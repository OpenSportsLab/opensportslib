"""Stream original FineGym videos, rebuild all spotting clips, and publish on success.

Byte-range prefetch bounds memory; only a small number of source videos are
stored concurrently. Source video files downloaded by this tool are removed
only after their clips pass RGB verification. Existing user media is untouched.
The complete benchmark is required before publication.
"""
import concurrent.futures,io,json,requests,tarfile,threading,time
import argparse,shutil,sys
from pathlib import Path

class RangedParts(io.RawIOBase):
 def __init__(self,parts,workers=16,chunk=8*1024*1024):
  self.parts=parts;self.chunk=chunk;self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=workers)
  self.workers=workers;self.pos=0;self.length=sum(x['size'] for x in parts);self.urls={};self.lock=threading.Lock();self.pending={};self.buffer=b'';self.cursor=0;self.next_index=0
  self.next_progress=time.monotonic()+30
 def _url(self,part,refresh=False):
  with self.lock:
   if part not in self.urls or refresh:
    r=requests.get('https://openxlab.org.cn/datasets/resolve/11/main/'+self.parts[part]['path'].lstrip('/'),allow_redirects=False,timeout=30)
    if r.status_code!=302:raise RuntimeError(f'Cannot resolve source part: {r.status_code}')
    self.urls[part]=r.headers['Location']
   return self.urls[part]
 def _fetch(self,index):
  global_start=index*self.chunk;part=0;local=global_start
  while part<len(self.parts) and local>=self.parts[part]['size']:
   local-=self.parts[part]['size'];part+=1
  if part==len(self.parts):return b''
  # Archive part sizes are aligned to this chunk size, except the last part.
  n=min(self.chunk,self.parts[part]['size']-local)
  for attempt in range(4):
   try:
    url=self._url(part,refresh=attempt>0)
    with requests.get(url,headers={'Range':f'bytes={local}-{local+n-1}'},stream=True,timeout=(30,120)) as r:
     if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {local}-{local+n-1}/{self.parts[part]["size"]}':
      raise RuntimeError(f'Incorrect range response: {r.status_code}')
     b=r.raw.read(n)
     if len(b)!=n:raise IOError('Short range response')
     return b
   except Exception as exc:
    print('range retry',index,attempt+1,type(exc).__name__,flush=True)
    if attempt==3:raise
    time.sleep(1)
 def read(self,n=-1):
  if n<0:n=self.chunk
  result=bytearray()
  while len(result)<n and self.pos<self.length:
   while len(self.pending)<self.workers and self.next_index*self.chunk<self.length:
    self.pending[self.next_index]=self.pool.submit(self._fetch,self.next_index);self.next_index+=1
   if self.cursor==len(self.buffer):
    index=self.pos//self.chunk
    self.buffer=self.pending.pop(index).result();self.cursor=0
   take=min(n-len(result),len(self.buffer)-self.cursor)
   result.extend(self.buffer[self.cursor:self.cursor+take]);self.cursor+=take;self.pos+=take
   if time.monotonic()>=self.next_progress:
    print('archive bytes read',self.pos,'/',self.length,flush=True)
    self.next_progress=time.monotonic()+30
  return bytes(result)
 def close(self):
  self.pool.shutdown(wait=True,cancel_futures=True);super().close()


def publish(output,release,repo,parent_commit):
 from huggingface_hub import HfApi,CommitOperationAdd,CommitOperationDelete
 from opensportslib.tools.osl_json_to_parquet import convert_json_to_parquet
 report=json.loads((output/'reports/build.json').read_text())
 if not report.get('complete') or report['clips']!=5374:
  raise ValueError('Refusing to publish an incomplete benchmark')
 for split in ['train','valid','test']:
  convert_json_to_parquet(output/f'{split}.json',output,release/split,
                          shard_size='1GB',missing_policy='raise',overwrite=True)
 shutil.copytree(output/'annotations',release/'annotations',dirs_exist_ok=True)
 shutil.copytree(output/'reports',release/'reports',dirs_exist_ok=True)
 shutil.copytree(output/'source-receipts',release/'source-receipts',dirs_exist_ok=True)
 card='''---
pretty_name: FineGym Precise Event Spotting
license: cc-by-nc-4.0
task_categories:
- video-classification
tags:
- gymnastics
- action-spotting
- precise-event-spotting
- opensportslib
configs:
- config_name: default
  data_files:
  - split: train
    path: train/metadata.parquet
  - split: validation
    path: valid/metadata.parquet
  - split: test
    path: test/metadata.parquet
---

# FineGym precise event spotting

Complete 5,374-clip spotting benchmark: 3,327 train, 944 validation and 1,103
test clips, with 32 event classes. FG-Start is evaluated by selecting the
upstream start-event classes from the same full-set predictions.

Original videos were obtained from OpenDataLab/FineGym. Some source encodings
are 720p where the reference metadata lists 1080p; measured and reference
dimensions are both recorded. All accepted sources exceed the model's input
resolution and preserve the reference aspect ratio. FPS is checked before
trimming. Both random temporal
padding and frame-rate stride follow the upstream spotting annotations.
Frames are resized to 224 by 224 with OpenCV linear interpolation and encoded
as RGB-lossless FFV1 Matroska. Every generated clip is fully decoded and its
RGB sequence checked against the resized source sequence. The resize and
lossless encoding are explicit conversion choices; source-video hashes,
padding, stride and upstream records are retained for reproducibility.

This release replaces an incomplete 422-clip conversion. An examined legacy
clip had 149 frames where the padded reference required 266; the old conversion
must not be used to reproduce the spotting benchmark. Repository history and
the original `raw` branch are retained.

## Format and loading

Each split has OSL `metadata.parquet`, `shard_manifest.parquet` and WebDataset
TAR shards. Download with OpenSportsLib's `download_dataset_split_from_hf`,
using repo `OpenSportsLab/OSL-loc-gymnastics`, revision `main`, split `test`
and `download_format="parquet"`. Authentication and accepted access conditions
are required. The source annotations in `annotations/` preserve original event
frame indices. Event timestamps use each sample's reference frame rate.

## Provenance and terms

Source dataset: https://sdolivia.github.io/FineGym/
Full source archive: https://opendatalab.com/OpenDataLab/FineGym
Spotting annotations: https://github.com/arturxe2/T-DEED/tree/COMMIT/data/finegym

See `reports/build.json`, `reports/archive-source.json` and `source-receipts/`.
The archive index records the publisher's part checksums; streaming may stop
once all required sources are obtained, so those full-part checksums are not
claimed as locally verified. Extracted source-video hashes are locally verified.
The original FineGym annotation license is CC BY-NC 4.0. Cite FineGym (Shao et
al., CVPR 2020), E2E-Spot (Hong et al., ECCV 2022), and T-DEED (Xarles et al.,
CVPR Workshops 2024). The existing manual access gate is preserved.
'''.replace('COMMIT',report['source_commit'])
 (release/'README.md').write_text(card)
 api=HfApi()
 info=api.dataset_info(repo)
 if info.sha!=parent_commit:
  raise RuntimeError('Repository changed during rebuild; refusing to overwrite concurrent changes')
 files=[p for p in sorted(release.rglob('*')) if p.is_file()]
 keep={p.relative_to(release).as_posix() for p in files}
 operations=[CommitOperationAdd(path_in_repo=p.relative_to(release).as_posix(),path_or_fileobj=str(p)) for p in files]
 operations.extend(CommitOperationDelete(path_in_repo=x.rfilename) for x in info.siblings
   if x.rfilename not in keep and (x.rfilename.startswith(('train/','valid/','test/','reports/'))
                                 or x.rfilename.startswith('annotations-localization-')))
 commit=api.create_commit(repo_id=repo,repo_type='dataset',revision='main',parent_commit=parent_commit,
                          operations=operations,commit_message='Replace incomplete conversion with full frame-verified FineGym benchmark')
 write_json(output/'reports/publication.json',{'repo':repo,'commit':commit.oid,'url':commit.commit_url})
 print(commit.commit_url,flush=True)
 from opensportslib.tools import download_dataset_split_from_hf,download_dataset_sample_inputs_from_hf
 from prepare_finegym_spotting import checksum
 checks=[]
 for split in ['train','valid','test']:
  result=download_dataset_split_from_hf(repo,commit.oid,split,str(output.parent/'hub-verification'),annotations_only=True)
  manifest=Path(result['json_path']);row=json.loads(manifest.read_text())['data'][0]
  selected=download_dataset_sample_inputs_from_hf(str(manifest),row['id'])
  if selected.get('failed_paths'):raise RuntimeError('Hub sample download failed')
  video=manifest.parent/row['inputs'][0]['path']
  if checksum(video)!=row['metadata']['sha256']:raise ValueError('Hub sample checksum mismatch')
  checks.append({'split':split,'sample':row['id'],'sha256':row['metadata']['sha256']})
  print('verified Hub split',split,flush=True)
 write_json(output/'reports/hub-verification.json',checks)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--reference',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--source-root',type=Path,required=True)
 p.add_argument('--source-commit',required=True)
 p.add_argument('--workers',type=int,default=8)
 p.add_argument('--publish',action='store_true')
 p.add_argument('--repo-id',default='OpenSportsLab/OSL-loc-gymnastics')
 a=p.parse_args()
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'convert'))
 global write_json
 from prepare_finegym_spotting import prepare_source,reference_rows,finalize,write_json
 from huggingface_hub import HfApi
 parent=HfApi().dataset_info(a.repo_id).sha if a.publish else None
 a.source_root.mkdir(parents=True,exist_ok=True)
 response=requests.get('https://openxlab.org.cn/datasets/api/v2/datasets/OpenDataLab%2CFineGym/r/main?recursive=true&needContent=true&limit=500',timeout=30)
 response.raise_for_status();inventory=response.json()
 parts=sorted([x for x in inventory['data']['list'] if x['path'].startswith('/raw/')],key=lambda x:x['path'])
 if not parts or any(x['size']%(8*1024*1024) for x in parts[:-1]):
  raise ValueError('Unexpected archive part boundaries')
 write_json(a.output/'reports/archive-source.json',inventory)
 expected={row['video'].split('_E_')[0] for _,row in reference_rows(a.reference)}
 complete=set()
 for path in (a.output/'source-receipts').glob('*.json'):
  receipt=json.loads(path.read_text())
  if receipt.get('complete') and receipt.get('source_commit')==a.source_commit:
   complete.add(receipt['source_video'])
 write_json(a.output/'reports/progress.json',{'sources_complete':len(complete),'sources_expected':len(expected),'status':'running'})
 futures={}
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)

 def process(path):
  result=prepare_source(path,a.reference,a.output,a.source_commit,workers=a.workers)
  path.unlink()  # Only files downloaded to the owned source staging directory.
  return result

 def finish_one():
  identity,future=next(iter(futures.items()))
  result=future.result();del futures[identity];complete.add(identity)
  write_json(a.output/'reports/progress.json',{'sources_complete':len(complete),'sources_expected':len(expected),
               'last_source':identity,'last_source_clips':result['clips'],'status':'running'})
  print('sources',len(complete),'/',len(expected),'completed',identity,flush=True)

 stream=RangedParts(parts)
 try:
  if complete!=expected:
   with tarfile.open(fileobj=stream,mode='r|gz',bufsize=1024*1024) as archive:
    for member in archive:
     if not member.isfile() or Path(member.name).suffix.lower() not in {'.mp4','.mkv','.webm'}:continue
     identity=Path(member.name).stem
     print('archive source',identity,member.size,flush=True)
     if identity not in expected or identity in complete or identity in futures:continue
     while len(futures)>=2:finish_one()
     if complete==expected:break
     if shutil.disk_usage(a.output.parent).free<member.size+20_000_000_000:
      raise RuntimeError('Insufficient disk space for the next source video')
     destination=a.source_root/(identity+Path(member.name).suffix)
     if not destination.exists():
      partial=destination.with_suffix(destination.suffix+'.stream.part')
      print('extract',identity,member.size,flush=True)
      with archive.extractfile(member) as src,partial.open('wb') as dst:
       shutil.copyfileobj(src,dst,length=1024*1024)
      partial.replace(destination)
     futures[identity]=pool.submit(process,destination)
     if len(complete)+len(futures)==len(expected):break
  while futures:finish_one()
  report=finalize(a.reference,a.output,a.source_commit)
  if a.publish:publish(a.output,a.output.parent/'release',a.repo_id,parent)
  write_json(a.output/'reports/progress.json',{'status':'complete',**report})
 except (Exception,KeyboardInterrupt) as e:
  write_json(a.output/'reports/progress.json',{'status':'interrupted' if isinstance(e,KeyboardInterrupt) else 'failed','error':str(e),
             'sources_complete':len(complete),'sources_expected':len(expected)})
  raise
 finally:
  stream.close();pool.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':
 main()
