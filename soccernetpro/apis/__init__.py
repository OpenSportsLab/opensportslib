# soccernetpro/apis/__init__.py

# Import task APIs
from soccernetpro.apis.classification import ClassificationAPI
from soccernetpro.apis.localization import LocalizationAPI

# Factory functions for user-facing calls
def classification(config=None, data_dir=None, save_dir=None):#, pretrained_model=None):
    return ClassificationAPI(config=config, data_dir=data_dir, save_dir=save_dir)#,pretrained_model=pretrained_model)

def localization(config=None, data_dir=None, save_dir=None):#, pretrained_model=None):
    return LocalizationAPI(config=config, data_dir=data_dir, save_dir=save_dir)#,pretrained_model=pretrained_model)


# Expose only these
__all__ = [
    "classification",
    "localization",
]
