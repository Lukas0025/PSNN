#!/usr/bin/env python3	
# -*- coding: utf-8 -*-	
## @package PSNN.server_tools
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# server tools

import requests

class server_tools:

    def __init__(self, model):
        self.modelsServer = model.modelsServer

    ## upload model to models server
    # 
    # @param object self
    # @param user - username
    # @param password - password for username
    # @param model - model what we want upload
    # @param name - name of model
    # @param description - description of model
    # @param loss - loss of model
    # @return json obj
    def upload(self, user, password, model, name, description, loss):
    
        # dump model to string
        smodel = model.dumps()

        data = {
            'a': 'uploadModel',
            'user': user,
            'pass': password,
            'model': smodel,
            'desc': description,
            'name': name,
            'loss': loss
        }

        r = requests.post(
            url = model.modelsServer,
            data = data
        )

        return r.json()

    ## get info about model
    #
    # @param self
    # @param name - full name of model
    # @return json obj
    def getModelInfo(self, name):
        data = {
            'a': 'getModelInfo',
            'name': name
        }

        r = requests.post(
            url = self.modelsServer,
            data = data
        )

        return r.json()

    ## register user
    # 
    # @param self
    # @param name - full name of model
    # @return json obj
    def register(self, name, password):
        data = {
            'a': 'register',
            'name': name,
            'pass': password
        }

        r = requests.post(
            url = self.modelsServer,
            data = data
        )

        return r.json()

    ## get all models with name like
    # 
    # @param self
    # @param name - full name of model
    # @return json obj
    def findModel(self, name):
        data = {
            'a': 'findModel',
            'name': name
        }

        r = requests.post(
            url = self.modelsServer,
            data = data
        )

        return r.json()