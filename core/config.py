# Application Settings
# From the enviornments vaiables
import os
from pydantic import Field
from pydantic_settings import BaseSettings

from pathlib import Path

class ApplicationSettings(BaseSettings):
    DEPLOYED_BASE_PATH: str = Field(alias='DEPLOYED_BASE_PATH')

class Settings(ApplicationSettings):
    PROJECT_NAME: str = 'Seeing Is Believivng'

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

class PathConfig:
    PARENT_PATH = Path.cwd().parent
    if 'sib' not in str(PARENT_PATH):
        PARENT_PATH = PARENT_PATH / 'sib'
    
    PARENT_PATH_STR = str(PARENT_PATH)

    TEMPLATE_DIRECTORY = os.path.join(PARENT_PATH_STR, 'templates')
    IMAGES_DIRECTORY = os.path.join(PARENT_PATH_STR, 'images')
    FILE_DIRECTORY = os.path.join(PARENT_PATH_STR, 'files')
    MODEL_DIRECTORY = os.path.join(PARENT_PATH_STR, 'models')
    SCRIPT_DIRECTORY = os.path.join(PARENT_PATH_STR, 'scripts')

    yolo_weightfile=os.path.join(FILE_DIRECTORY,  'yolov3.weights')
    yolo_classesfile=os.path.join(FILE_DIRECTORY,  'yolov3.txt')
    yolo_configfile=os.path.join(FILE_DIRECTORY,  'yolov3.cfg')
    haat_caascade_file = os.path.join(FILE_DIRECTORY,'haarcascade_frontalface_default.xml')

    plant_classifier_model_path = os.path.join(MODEL_DIRECTORY,'plantclassifier.model')
    ultralytics_sib_local_model_path = os.path.join(MODEL_DIRECTORY,'sib-model.pt')

    @classmethod
    def init_app(cls, app):
        app.template_folder = cls.TEMPLATE_DIRECTORY

settings = Settings()
