# opensportslib/apis/__init__.py

# Import task APIs
from opensportslib.apis.classification import ClassificationAPI
from opensportslib.apis.localization import LocalizationAPI
import warnings
warnings.filterwarnings("ignore")

# Factory functions for user-facing calls
def classification(
    config=None,
    data_dir=None,
    save_dir=None,
    dataset_repo_id=None,
    dataset_revision=None,
    hf_token=None,
):#, pretrained_model=None):
    return ClassificationAPI(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        hf_token=hf_token,
    )#,pretrained_model=pretrained_model)

def localization(
    config=None,
    data_dir=None,
    save_dir=None,
    dataset_repo_id=None,
    dataset_revision=None,
    hf_token=None,
):#, pretrained_model=None):
    return LocalizationAPI(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        hf_token=hf_token,
    )#,pretrained_model=pretrained_model)


# Expose only these
__all__ = [
    "classification",
    "localization",
]
