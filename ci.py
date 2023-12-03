import os


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


IMAGE = 'jeancarlomachado/social_good_detector:latest'
def push_ecr():
    run_cmd(f"docker push {IMAGE}")

def build_and_push():
    run_cmd(f"docker build -t {IMAGE} .")
    push_ecr()



if __name__ == "__main__":
    import fire
    fire.Fire()