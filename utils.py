import yaml

def loadConfig(path="./config/conf.yaml"):
    with open(path, 'r') as yml:
        config = yaml.full_load(yml)
        yml.close()
        
    return config