# SDK模型下载
from re import M
from modelscope import snapshot_download
from modelscope.hub.api import HubApi
from pathlib import Path

ms_token = "ms-e0addc95-e8dd-459c-b5b8-4021df169453"


ms_repo = "chenda/Moonlight-16B-A3B-Instruct-RTN-dsf8_w8a8"
local_dir = "/home/admin/csl/code/llmc/checkpoints/Moonlight-16B-A3B-Instruct/RTN/dsf8_w8a8"

api = HubApi()
api.login(ms_token)
Path(local_dir).mkdir(exist_ok=True, parents=True)
model_dir = snapshot_download(ms_repo, local_dir=local_dir)
