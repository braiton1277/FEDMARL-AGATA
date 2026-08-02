import sys
import os
import json
from mi.rpc_client import rpc_send
import subprocess

class FTDL:
    def __init__(self, side:str="client") -> None:
        self.task_id = os.environ["TASK_ID"]
        m = {"client":"CLIENT","server":"SERVER"}
        self.broker_host = os.environ[f"{m[side]}_BROKER_HOST"]
        self.broker_port = os.environ[f"{m[side]}_BROKER_PORT"]
        print(self.task_id," ",self.broker_host,":",self.broker_port)
        
    def _send_to_broker(self, func, json_message):
        received = rpc_send(
            func,
            request=json_message,
            host=self.broker_host,
            port=self.broker_port)
        print("Response: ", received)

    def send_stats(
        self, 
        new_round:bool=False, 
        acc_contrib:int=None,
        data_qnt:int=None,
        complement_dict:dict={}
    ): 
        message = complement_dict
        message["type"] = "stats"
        message["acc_contrib"] = acc_contrib
        message["new_round"] = new_round
        message["data_qnt"] = data_qnt

        self._send_to_broker("rpc_exec_handle_stats_from_task", message)
        
    def announce_finish(self):
        self._send_to_broker("rpc_exec_handle_finish_from_task",
                             {"type":"finished","task_id": self.task_id})

    def announce_error(self,exception:Exception):
        self._send_to_broker("rpc_exec_handle_error_from_task",
            {"type":"error","task_id": self.task_id,"exception": str(exception)})

    def trigger(self, trigger_name:str, trigger_arguments:list=[]):
        execution_command_list = [sys.executable, trigger_name]
        execution_command_list.extend(trigger_arguments)
        self.process = subprocess.Popen(execution_command_list)