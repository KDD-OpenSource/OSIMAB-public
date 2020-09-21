
import yaml

class Dummy:
    def __init__(*args, **kwargs):
        pass

class config:
    def __init__(self, external_path=None):

        if external_path:
            stream = open(external_path, "r")
            docs = yaml.safe_load_all(stream)
            for doc in docs:
                for k, v in doc.items():
                    cmd = "self."+k+"=Dummy()"
                    exec(cmd)
                    # if k == "train":
                    if type(v) is dict:
                        for k1, v1 in v.items():
                            cmd = "self."+k+"." + k1 + "=" + repr(v1)
                            print(cmd)
                            exec(cmd)
                    else:
                        cmd = "self."+k+"="+repr(v)
                        print(cmd)
                        exec(cmd)
            stream.close()
