import os
from ftdl.task_exceptions import *
#import docker
from python_on_whales import DockerClient, docker
import subprocess
import time
import threading

class Task:
    """
    A Task is an interface for running code as a container
    This is a base class, inherited by ClientSideTask and ServerSideTask

    :param work_path: project location, within which "tasks" dir resides
    :type work_path: str

    :param task_id: hexadecimal number identifying task
    :type task_id: str

    :param side: server or client
    :type side: str

    :raises FileNotFoundError: no directory named "{work_path}/tasks/task_{task_id}"
    """
    def __init__(
            self, 
            work_path: str, 
            task_id: str, 
            side:str, 
            host:str) -> None:
        self.work_path = work_path
        self.task_id = task_id
        self.task_dir_name = self._get_task_directory_name(task_id)
        #self.dockerfile = self._get_dockerfile_name(self.task_dir_name, side)
        self.composefile = self._get_composefile_name(self.task_dir_name, side)
        self.running = False
        #self.docker_handler = docker.from_env()
        self._run_config_script(side)
        os.environ["DOCKER_HOST"] = host
        self.docker_handler = DockerClient(
            compose_files=[self.composefile])
        self.started = False 
        self._load_image_if_exists(side=side)

    def _get_task_directory_name(self, task_id:str):
        task_dir_name = os.path.join(self.work_path, f"tasks/task_{task_id}")
        if not os.path.isdir(task_dir_name):
            raise FileNotFoundError(f"Directory '{task_dir_name}' does not exist.")
        return str(task_dir_name)

    def _run_config_script(self,side):
        ftdl_path = os.path.join(self.work_path,'FTDL')
        subprocess.Popen(["cp","-r",ftdl_path,
                          os.path.join(self.task_dir_name,side)
                          ])
        time.sleep(5)

    def _get_composefile_name(self, task_dir_name:str, side:str): 
        file_name = os.path.join(task_dir_name, f'{side}/compose.yaml')
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File '{file_name}' not found in directory '{task_dir_name}'.")
        return file_name

    def _load_image_if_exists(self,side):
        image_tar_path = os.path.join(self.task_dir_name,f"{side}/image.tar")
        if os.path.isfile(image_tar_path):
            print(f"Loading docker image from {image_tar_path}...")
            self.docker_handler.load(image_tar_path)
        self.build = not os.path.isfile(image_tar_path)

    def up(self):
        self.docker_handler.compose.up(build=self.build, detach=True)
        self.started = True

    def down(self):
        self.docker_handler.compose.down()
        self.started = False

    def run_task(self, backgorund=True):
        if self.started:
            raise TaskAlredyRunning(self.task_id)
        if backgorund:
            t = threading.Thread(target=self.up)
            t.start()
        else:
            self.up()

    def stop_task(self, backgorund=True):
        if not self.started:
            raise TaskAlredyStopped(self.task_id)
        if backgorund:
            t = threading.Thread(target=self.down)
            t.start()
        else:
            self.down()
