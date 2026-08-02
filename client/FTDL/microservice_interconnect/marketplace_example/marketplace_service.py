import json
import os
from mi.base_service import BaseService
from pathlib import Path

class MarketplaceService(BaseService):
    def __init__(self, work_path:str, broker_host:str, broker_port:int):
        super().__init__(broker_host=broker_host, broker_port=broker_port, work_path=work_path)

        self.add_endpoint(self.rpc_exec_run_sql)
        self.add_endpoint(self.rpc_exec_get_model_from_name_version)
        self.add_endpoint(self.rpc_exec_insert_model)
        self.add_endpoint(self.rpc_exec_search_for_tags)
        self.add_endpoint(self.rpc_exec_update_model)
        self.add_endpoint(self.rpc_exec_download_training_files)
        self.add_endpoint(self.rpc_exec_download_inference_files)

    def rpc_exec_run_sql(self, received:dict)->dict:
        """
        Receives and execute raw SQL command. Used by DIS LLM

        :param received: JSON containing 'query'
        :type received: dict

        :raises sqlite3.IntegrityError: could not perform DB statement

        :return: dictionary representing all returned models
        :rtype: dict
        """
        raise NotImplementedError

    def rpc_exec_get_model_from_name_version(self, received:dict)->dict:
        """
        Receives validated JSON and query model. 

        :param received: JSON containing model 'name' and (optionaly) 'version' 
        :type received: dict

        :raises sqlite3.IntegrityError: could not perform DB statement

        :return: dictionary representing the model
        :rtype: dict
        """
        raise NotImplementedError

    def rpc_exec_insert_model(self, received:dict):
        """
        Insert model using given dictionary representing it. 

        :param received: JSON representing model (look at the schema)
        :type received: dict

        :raises sqlite3.IntegrityError: could not perform DB statement
        """
        raise NotImplementedError

    def rpc_exec_insert_model(self, received:dict):
        """
        Insert model using given dictionary representing it. 

        :param received: JSON representing model (look at the schema)
        :type received: dict

        :raises sqlite3.IntegrityError: could not perform DB statement
        """
        raise NotImplementedError
    
    def rpc_exec_search_for_tags(self, received:dict)->dict:
        """
        Search model mathcing tags 

        :param received: JSON with 'tags_list'
        :type received: dict

        :raises sqlite3.IntegrityError: could not perform DB statement
        
        :return: dictionary representing all models
        :rtype: dict
        """
        raise NotImplementedError
    
    def rpc_exec_update_model(self, received:dict)->dict:
        """
        Search model mathcing tags 

        :param received: JSON with 'tags_list'
        :type received: dict

        :raises sqlite3.IntegrityError: could not perform DB statement
        
        :return: dictionary representing all models
        :rtype: dict
        """
        raise NotImplementedError
    
    def rpc_exec_download_training_files(self, received:dict)->dict:
        """
        Return files used for training (dockerfile, compose, python, models in PT/ONNX, etc)

        :param received: JSON with 'task_id'
        :type received: dict

        :raises sqlite3.IntegrityError: could not perform DB statement
        
        :return: dictionary representing with each filename/path as key and the corresponding base64 as value
        :rtype: dict
        """
        raise NotImplementedError
    
    def rpc_exec_download_inference_files(self, received:dict)->dict:
        """
        Return files used for serve an inference service at the client 

        :param received: JSON with 'task_id'
        :type received: dict

        :raises sqlite3.IntegrityError: could not perform DB statement
        
        :return: dictionary representing with each filename/path as key and the corresponding base64 as value. 
        :rtype: dict
        """
        raise NotImplementedError



if __name__ == "__main__":
    print("Starting microservice")
    service = MarketplaceService(str(Path().resolve()), "localhost", 5000)
    service.start()
