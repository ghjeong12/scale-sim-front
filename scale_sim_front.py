'''
Copyright (c) 2020 Georgia Instititue of Technology
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author : Geonhwa Jeong (geonhwa.jeong@gatech.edu)
'''

import re
import argparse
from argparse import RawTextHelpFormatter

if __name__ == "__main__":
    target_sim = 'scale-sim'

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--api_name', type=str, default="pytorch", help="api choices: pytorch, keras")
    parser.add_argument('--input_size', type=str, default="3,224,224", help='input size')
    parser.add_argument('--model', type=str, default="mobilenet_v2",
    help='model from torchvision choices: \n'
         'resnet18, alexnet, vgg16, squeezenet, densenet, \n'
         'inception_v3, googlenet, shufflenet, \n'
         'mobilenet_v2, wide_resnet50_2, mnasnet,\n'
         '-----\n'
         'model from tensorflow.keras.applications choices: \n'
         'xception, vgg16, vgg19, resnet50, resnet101, \n'
         'resnet152, resnet50_v2, resnet101_v2, resnet152_v2, \n'
         'inception_v3, inception_resnet_v2, mobilenet, mobilenet_v2,\n'
         'densenet121, densenet169, densenet201, nasnet_large, \n'
         'nasnet_mobile\n'
         '-----\n'
         'To use a custom model, enter custom for this arguement')
    parser.add_argument('--custom', type=str, default="none",
    help='Enter the custom network python file name here.\n'
         'The file should have a function with same file name\n '
         'which returns the model\n'
         '(This option is working only for keras)\n')


    parser.add_argument('--dataflow', type=str, default="os", help='dataflow choices: dla, os, ws, rs')
    parser.add_argument('--outfile', type=str, default="out.m", help='output file name')
    opt = parser.parse_args()
    INPUT_SIZE = tuple((int(d) for d in str.split(opt.input_size, ",")))

    print('Begin processing')
    print('API name: ' + str(opt.api_name))
    print('Model name: ' + str(opt.model))
    print('Input size: ' + str(INPUT_SIZE))
    if(opt.api_name =='keras'):
        from keras_helper import get_model
        from keras_maestro_summary import summary

        model = None
        if opt.model == 'custom':
            new_module = __import__(opt.custom)
            model = getattr(new_module, opt.custom)()
        else:
            model = get_model(opt.model, INPUT_SIZE[::-1])

        mae_summary = summary(model)

        with open("out/"+opt.outfile, "w") as fo:
            fo.write("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,\n")
            for key, val in mae_summary.items():
                pc = re.compile("^CONV")
                pd = re.compile("^DSCONV")
                pf = re.compile("^Dense")

                match_pc = pc.match(val['type'])
                match_pd = pd.match(val['type'])
                match_pf = pf.match(val['type'])

                if match_pc:
                    if(key == 'conv2d'):
                        fo.write('conv2d_0, ')
                    else:
                        fo.write("{}, ".format(key))
                    #type = val["type"]
                    #fo.write("Type: {}\n".format(type))
                    ifmap_height = val['dimension_ic'][5]
                    ifmap_width = val['dimension_ic'][6]
                    filter_height = val['dimension_ic'][3]
                    filter_width = val['dimension_ic'][4]
                    channels = val['dimension_ic'][2]
                    num_filter = val['dimension_ic'][1]

                    assert(val['strides'][0] == val["strides"][1])
                    strides = val['strides'][0]
                    fo.write("{}, \t{}, \t{}, \t{}, \t{}, \t{}, \t{},\n".format(ifmap_height, ifmap_width, filter_height, filter_width, channels, num_filter, strides))
                    '''
                    fo.write("Dimensions {{ K: {}, C: {}, R: {}, S: {}, Y: {}, X: {} }}\n".format(
                        *val["dimension_ic"][1:]))

                    fo.write("Stride {{ X: {}, Y: {} }}\n".format(*val["strides"]))
                    '''
    if(opt.api_name == 'pytorch'):
        import torch
        import torchvision.models as models
        from torch_maestro_summary import summary

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = getattr(models, opt.model)()
        model = model.to(device)
        mae_summary = summary(model, INPUT_SIZE)
        with open(opt.dataflow + ".m", "r") as fd:
            with open("util/dpt.m", "r") as fdpt:
                with open("out/"+opt.outfile, "w") as fo:
                    fo.write("Network {} {{\n".format(model.__module__))
                    for key, val in mae_summary.items():
                        pc = re.compile("^Conv")
                        pl = re.compile("^Linear")
                        match_pc = pc.match(key)
                        match_pl = pl.match(key)
                        if match_pc or match_pl:
                            fo.write("Layer {} {{\n".format(key))
                            type = val["type"]
                            fo.write("Type: {}\n".format(type))
                            if not match_pl:
                                fo.write("Stride {{ X: {}, Y: {} }}\n".format(*val["stride"]))
                            fo.write("Dimensions {{ K: {}, C: {}, R: {}, S: {}, Y: {}, X: {} }}\n".format(
                                *val["dimension_ic"][1:]))
                            if type == "CONV":
                                fd.seek(0)
                                fo.write(fd.read())
                            else:
                                fdpt.seek(0)
                                fo.write(fdpt.read())
                            fo.write("}\n")
                    fo.write("}")

    print("Done converting for %s" % target_sim)
