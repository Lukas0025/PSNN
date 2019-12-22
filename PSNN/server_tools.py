#!/usr/bin/env python3	
# -*- coding: utf-8 -*-	
## @package server_tools.py
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
    # @param str user - username
    # @param str password - password for username
    # @param object model - model what we want upload
    # @param str name - name of model
    # @param str description - description of model
    # @param float loss - loss of model
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
    # @param object self
    # @param str name - full name of model
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
    # @param object self
    # @param str name - full name of model
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
    # @param object self
    # @param str name - full name of model
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