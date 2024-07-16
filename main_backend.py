from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_from_directory, abort
import json
app = Flask(__name__)
from lib.utils_serve import model_edit, plot_imgs, edit_status_check, get_database
import os
import argparse
from PIL import Image
from lib.utils_config import exp_config_gen
from lib.policy_manager import ModeratorPolicyManager

policy_manager = ModeratorPolicyManager()

@app.route('/index')
def index():
    return render_template('interface-2.html')

@app.route('/pretrain_img_generate', methods=['POST'])
def pretrain_img_generate():
    config_data = request.get_data()
    config_data =  json.loads(config_data)
    prompt=config_data["prompt"]
    pretrain_image_list = policy_manager.call_pretrain_model(
        prompt=prompt
    )
    return pretrain_image_list

@app.route('/get_all_policies', methods=['GET'])
def get_all_policies():
    policy_name_list = policy_manager.get_all_policies()
    return jsonify(policy_name_list)

@app.route('/get_policy_by_name', methods=['POST'])
def get_policy_by_name():
    config_data = request.get_data()
    config_data =  json.loads(config_data)
    policy_name = config_data["policy_name"]
    return jsonify(policy_manager.get_policy(policy_name=policy_name))

@app.route('/img_generate', methods=['POST'])
def img_generate():
    config_data = request.get_data()
    config_data =  json.loads(config_data)
    
    prompt=config_data["prompt"]
    policy_name=config_data["policy_name"]
    
    edited_image_list = policy_manager.call_policy_model(
        policy_name=policy_name,
        prompt=prompt
    )
    return edited_image_list
    
@app.route('/model_edit', methods=['POST'])
def edit():
    config_data = request.get_data()
    config_data =  json.loads(config_data)
    
    policy_dict = config_data["policy_dict"]
    policy_name = config_data["policy_name"]
    
    edit_flag = policy_manager.craft_policy(
        policy_dict=policy_dict,
        policy_name=policy_name
    )
    return edit_flag

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7417)