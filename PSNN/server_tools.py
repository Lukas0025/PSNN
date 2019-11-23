#!/usr/bin/env python	
# -*- coding: utf-8 -*-	
#	
#  PSNN.py
#  server tools	
#  	
#  Copyright 2019 Lukáš Plevač <lukasplevac@gmail.com>	
#  	
#  This program is free software; you can redistribute it and/or modify	
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#

import requests

class server_tools:

  def __init__(self, model):
      self.modelsServer = model.modelsServer

  '''
  upload model to models server

  @param object self
  @param str user - username
  @param str password - password for username
  @param object model - model what we want upload
  @param str name - name of model
  @param str description - description of model
  @param float loss - loss of model
  @return json obj
  '''
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

  '''
  get info about model

  @param object self
  @param str name - full name of model
  @return json obj
  '''
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

  '''
  register user

  @param object self
  @param str name - full name of model
  @return json obj
  '''
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

  '''
  get all models with name like

  @param object self
  @param str name - full name of model
  @return json obj
  '''
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