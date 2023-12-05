import os


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


DOCKER_IMAGE = 'jeancarlomachado/social_good_detector:latest'
ECR_IMAGE = '467863034863.dkr.ecr.eu-central-1.amazonaws.com/social-good-detector-private:latest'

LOCAL_NAME = 'social_good_detector:latest'

def push_ecr():
    build(ECR_IMAGE)
    run_cmd(f"docker push {ECR_IMAGE}")

def build(tag=LOCAL_NAME):
    run_cmd(f"docker build --platform=linux/amd64 -t {tag} .")
def build_arm(tag=LOCAL_NAME):
    run_cmd(f"docker build  -t {tag} .")

def build_and_push():
    build()
    push_ecr()

def tag_and_push():
    run_cmd(f"docker tag {LOCAL_NAME} {ECR_IMAGE}")
    run_cmd(f"docker push {ECR_IMAGE} ")
def build_and_run():
    build_arm()
    run()

def run():
    run_cmd(f"docker run -p 8080:80 {LOCAL_NAME}")

if __name__ == "__main__":
    import fire
    fire.Fire()