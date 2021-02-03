import yaml


class Dummy:
    def __init__(*args, **kwargs):
        #import pdb; pdb.set_trace()
        pass

    def keys(self):
        return self.__dict__.keys()

    def dict(self):
        return self.__dict__

    def values(self):
        return self.__dict__.values()

class config:
    def __init__(self, external_path=None):

        if external_path:
            stream = open(external_path, "r")
            docs = yaml.safe_load_all(stream)
            self.config_dict = {}
            for doc in docs:
                for k, v in doc.items():
                    #import pdb; pdb.set_trace()
                    cmd = "self." + k + "=Dummy()"
                    exec(cmd)
                    if type(v) is dict:
                        for k1, v1 in v.items():
                            #import pdb; pdb.set_trace()
                            cmd = "self." + k + "." + k1 + "=" + repr(v1)
                            exec(cmd)
                    else:
                        cmd = "self." + k + "=" + repr(v)
                        exec(cmd)
                self.config_dict = doc
            stream.close()


# we need to recursively add Dummy(). Also each dummy must have attributes if
# values are dummies, or attributes values
