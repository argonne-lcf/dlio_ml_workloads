#!/usr/bin/python
import argparse
import json
import logging
import os
import re
import requests
from pathlib import Path
from pprint import pformat
from requests.auth import HTTPBasicAuth
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Setup logger
debug = bool(int(os.getenv("DEBUG", '0')))
logging_level = logging.DEBUG if debug else logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging_level)
LOGGER = logging.getLogger(__name__)

# Setup session
SESSION = requests.Session()
HEADERS = { 'PRIVATE-TOKEN' : os.getenv('JOB_LOG_DOWNLOAD_TOKEN') }

def main(output_dir):
    # Define endpoint URLs
    # For more details see https://docs.gitlab.com/ee/api/api_resources.html#project-resources
    project_endpoint, pipeline_endpoint, pipeline_job_endpoint = get_endpoints()

    LOGGER.info(f"Getting build jobs with 'failed/success' status.")
    list_pipeline_jobs_endpoint = f"{pipeline_job_endpoint}?scope[]=success&scope[]=failed&per_page=100"

    try:
        build_jobs = get_build_jobs(list_pipeline_jobs_endpoint)
    except:
        raise

    project_job_endpoint = f"{project_endpoint}/jobs"
    for job in build_jobs:
        job_log_url = f"{project_job_endpoint}/{job['id']}/trace"
        LOGGER.info(f"Getting library versions from {job['name']} logs")
        LOGGER.info(f"\tURL: {job_log_url}")
        create_update_lib_json_file(job_log_url, output_dir, job['name'])
    return

def get_endpoints():
    ci_api_v4_url = os.getenv("CI_API_V4_URL", "https://gitlab-master.nvidia.com/api/v4")
    ci_project_id = os.getenv('CI_PROJECT_ID', '9530')
    ci_pipeline_id = os.getenv('CI_PIPELINE_ID', '4155813')
    project_endpoint = f"{ci_api_v4_url}/projects/{ci_project_id}"
    pipeline_endpoint = f"{project_endpoint}/pipelines/{ci_pipeline_id}"
    pipeline_job_endpoint = f"{pipeline_endpoint}/jobs"
    return project_endpoint, pipeline_endpoint, pipeline_job_endpoint

def get_build_jobs(endpoint):
    LOGGER.debug(f"Getting all jobs.")
    resp = requests.get(endpoint, headers=HEADERS)
    resp.raise_for_status()
    jobs = resp.json()

    while 'next' in resp.links:
        LOGGER.debug('Getting next page:')
        resp = requests.get(resp.links['next']['url'], headers=HEADERS)
        resp.raise_for_status()
        jobs += resp.json()

    if LOGGER.isEnabledFor(logging.DEBUG):
        for j in jobs: LOGGER.debug(pformat(j))

    is_build_job = lambda job: job['stage'] == 'build'
    return filter(is_build_job, jobs)

def create_update_lib_json_file(job_log_url, output_dir, job_name = "none"):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = download_job_log(job_log_url)

    formatted_job_name = job_name.replace(" ","_").replace("/","-")
    job_log_path = output_dir / f"{formatted_job_name}-update_libs.log"
    job_log_path.write_text(data, encoding='utf-8')

    build_details = {}
    with job_log_path.open(encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('$ docker inspect -f "{{ .Config.Env }}"'):
                build_details.update(get_library_versions(next(fh)))
            elif line.startswith("FRAMEWORK_COMMIT_ID="):
                build_details['s_framework_commit_id'] = line.strip().split("FRAMEWORK_COMMIT_ID=")[1]
            elif line.startswith("FRAMEWORK_COMMIT_PREV="):
                build_details['s_framework_commit_prev'] = line.strip().split("FRAMEWORK_COMMIT_PREV=")[1]
            elif line.startswith("IMAGE_CREATION_TIME_IN_EPOCH="):
                build_details['ts_created'] = line.strip().split("IMAGE_CREATION_TIME_IN_EPOCH=")[1]

    json_output_path = job_log_path.with_suffix(".json")
    LOGGER.info(f"\tSaving as: {json_output_path}")
    json_output_path.write_text(json.dumps(build_details))

def download_job_log(url):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    resp = socket_connection(url)
    return ansi_escape.sub('',resp.text)

def get_library_versions(line):
    docker_env_vars = line.strip()[1:-1].split(' ')
    lib_versions = {
        f"s_env_{name.lower()}" : value
        for name,value in (env_vars.split('=',1)
        for env_vars in docker_env_vars if "=" in env_vars)
    }
    lib_versions['s_cudnn_version'] = lib_versions['s_env_cudnn_version']
    return lib_versions

def socket_connection(raw_url):
    for retry in range(3):
        try: return SESSION.get(raw_url, headers=HEADERS, verify=False, stream=True)
        except: LOGGER.info("retrying url ", raw_url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    main(Path(args.logdir))
