# ANN
neurons  = []
#gonna make a list of lists(each list being a row of neurons), with the second list being a list of dicts,
#containing an id and bias, then theres a list of id to id connections and the associated weight(weight[first id][second id])
#rows, cols = (5, 5)
#arr = [[0]*cols]*rows
from PIL import Image, ImageFilter
import copy
import psutil
import pickle
import random
import math
import csv
import numpy
#nimport skimage.measure
def sigmoid(x):

    exp = numpy.exp(-x)
    if exp > 1000000:
        return 0.0

    return 1.0 / (1.0 + exp)
    return 1 / (1 + math.exp(-x))
def load_image(path):

    img = Image.open(path)
    img = img.resize((20, 20))
    img = img.convert('L')
    img = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0))

    pixels = []
    image = []
    #for pixel in list(img.getdata()):
    #    pixels.append(pixel)
    #    if len(pixels) == 26:
    #        image.append(pixels)
    #        pixels.clear()


    img_width = 16
    img_height = 16

    pixels = []
    input_count = 0
    #for pixel in list(img.getdata()):
    #    input_count += 1
    #    av = 0
    #    for p in pixel:
    #        av = av + p
    #    av = round(av/765, 4)
    #    pixels.append(av)

    return list(img.getdata())
#pixels is list of all pixels in grayscale from 0-1

#neurons_inlayer = 14**2 #900
#neurons_layer1 = 100
#neurons_outlayer = 10
def create_random_network():

    layers = neuron_count

    #preparing space for structure of network
    network = []

    p = 0
    for i in range(len(layers)):
        p += 1
        c = 0
        this_layer = []
        
        for j in range((layers[i])):
            #following only applies if there is only that one input  layer
            if i != 0:
                this_layer.append({'bias': round(random.uniform(-0.25, 0.25), 3), 'layer': p, 'weights': [round(random.uniform(-0.25, 0.25), 3) for m in range(layers[i - 1])]})
            else:
                this_layer.append({'bias': round(random.uniform(-0.25, 0.25), 3), 'layer': p})

        network.append(this_layer)
    
    return network
    
def create_empty_network():
    layers = neuron_count

    #preparing space for structure of network
    network = []
   
    p = 0
    for i in range(len(neuron_count)):
        p += 1
        c = 0
        this_layer = []
        
        for j in range((neuron_count[i])):
            
            #following only applies if there is only that one input  layer
            if i != 0:
                this_layer.append({'bias': 0, 'layer': p, 'weights': [0 for m in range(neuron_count[i - 1])]})
                
            

        network.append(this_layer)
    #for t in range(10):
    #    if t != 0:
    #        print(len(network[t][5]["weights"]))
    return network


def save_network(path, network):
    with open(path, 'wb') as f:
        pickle.dump(network, f)
def load_network(path):
    with open(path, 'rb') as f:
        mynewlist = pickle.load(f)
    return mynewlist

def randomize_network(weightof, network, from_numb, to_numb):
    s=0
    for weight in weightof:
        s+=1
        if s > 100:
            for w in range(len(weight)):
                weight[w] = round(random.uniform(from_numb, to_numb), 3)
    a = 0
    for net in network:
        a += 1
        if a == 2:
            for n in net:
                n["bias"] = round(random.uniform(from_numb, to_numb), 3)

    return weightof, network



def ol_train_network_batch(network, learning_rate):
    




    cost_net = []
    for i in range(5):
        cost_net.append(create_empty_network())
    print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
    new_network = 0
    total_error = []
    indiv_highest=[]
    pop = []
    for i in range(6):
        pop.append(i)
    indiv_highest.append(pop)
    for ö in range(5400):

        learning_rate -= learning_rate/1000
        
        
        cost_net.append(create_empty_network())
        total_error = []
        lool = []
        a=0
        for i in range(10):

           
            path = str(i) + "-" + str(ö)
            o = test_network(network, path)

            layer1_output = o[0] 
            outlayer_output= o[1] 
            highest = o[2] 
            #print(highest)
            #print(layer1_output)
            #print(outlayer_output)


            local_error = []
            if highest not in lool:
                lool.append(highest)
            if highest == i:
                    a+=1
            for d in range(neurons_layer1):
                cost_net[5][1][d]['layer'] = 0
            for p in range(len(outlayer_output)):
                if p == i:
                    r = (outlayer_output[p]-1)
                else:
                    r = (outlayer_output[p]-0)
                local_error.append(r**2)
            w = (sum(local_error)/len(local_error))#/2#/len(indiv_highest[0])#NOT SURE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            total_error.append(sum(local_error)/len(local_error))
            for o in range(len(outlayer_output)):
                if o != i:
                    sigmar = (outlayer_output[o]-0)*outlayer_output[o]*(1-outlayer_output[o])
                else:
                    sigmar = (outlayer_output[o]-1)*outlayer_output[o]*(1-outlayer_output[o])
                print(outlayer_output[o], sigmar)
                cost_net[5][2][o]['bias']+=sigmar*local_error[o]
                for v in range(len(layer1_output)):
                    cost_net[5][2][o]['weights'][v] += sigmar*layer1_output[v]*local_error[o]
                    cost_net[5][1][v]['layer'] += sigmar*network[2][o]["weights"][v]
            for k in range(len(layer1_output)):
                sigmar = cost_net[5][1][k]['layer']*layer1_output[k]*(1-layer1_output[k])
                cost_net[5][1][k]["bias"] += sigmar*w
                for x in range((neurons_inlayer)):
                    cost_net[5][1][k]["weights"][x]+=sigmar*int(img[path][x])*w
        indiv_highest.append(lool)
        indiv_highest.pop(0)
        print(indiv_highest[0], a)
        #20bcouse**4#*len(indiv_highest)                  !!!! change to taking the len of previous indiv heigest, and multiplying it uptop !!!!
        #weight updates
        for g in range(neurons_layer1):
            network[1][g]["bias"] -= learning_rate*cost_net[5][1][g]["bias"]/10#normal weight update#
            for y in range(5): #momentum
                network[1][g]["bias"] -= learning_rate*((cost_net[4-y][1][g]["bias"]/10)/(y+1.5))
            for h in range(neurons_inlayer):
                network[1][g]["weights"][h]-=learning_rate*cost_net[5][1][g]["weights"][h]/10
                for b in range(5):
                    network[1][g]["weights"][h] -= learning_rate*((cost_net[4-b][1][g]["weights"][h]/10)/(b+1.5))
        for c in range(neurons_outlayer):
            network[2][c]["bias"]-=learning_rate*cost_net[5][2][c]["bias"]/10
            for y in range(5):
                network[1][c]["bias"] -= learning_rate*((cost_net[4-y][1][c]["bias"]/10)/(y+1.5))
            for j in range(neurons_layer1):
                network[2][c]["weights"][j]-=learning_rate*cost_net[5][2][c]["weights"][j]/10
                for b in range(5):
                    network[1][c]["weights"][j] -= learning_rate*((cost_net[4-b][1][c]["weights"][j]/10)/(b+1.5))
        print(cost_net[5][2][0])
        print(ulfrknevn)
        total_error_av = (sum(total_error)/len(total_error))
        if ö % 600 == 0:
            print(total_error_av)
            print(learning_rate)
        cost_net.pop(0)

    return network


def train_network_batch(network, learning_rate):
    cost_net = []
    for i in range(5):
        cost_net.append(create_empty_network())
    print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
    new_network = 0
    total_error = []
    indiv_highest=[]
    pop = []
    for i in range(6):
        pop.append(i)
    indiv_highest.append(pop)
    right = 0
    wrong = 0
    for ö in range(5400):
        #ALR:
        learning_rate -= learning_rate/1000
        
        
        cost_net.append(create_empty_network())
        total_error = []
        lool = []
        a=0
        weight_total = 0
        #print(len(network))
        for u in network[layer_count-2]: #and network[2]:
            for p in u["weights"]:
                weight_total += abs(p)#**2
                #print(len(u["weights"]))
        
        for i in range(10):

           
            path = str(i) + "-" + str(ö)

            #layer1_output, outlayer_output, highest = test_network(network, path)
            output = test_network(network, path)
            local_error = []
            #print(len(output), layer_count)
            if output[layer_count-1] not in lool:
                lool.append(output[layer_count-1])
            if output[layer_count-1] == i:
                    a+=1
                    right += 1
            else:
                wrong += 1
            for u in cost_net[5]:
                for d in range(len(u)):
                    u[d]['layer'] = 0
            if activation == "sigmoid":
                for p in range(len(output[layer_count-2])): #could use  .index()
                    if p == i:
                        r = (output[layer_count-2][p]-1)
                    else:
                        r = (output[layer_count-2][p]-0)
                    local_error.append(r**2)
            elif activation == "softmax":
                local_error.append(-(math.log(output[layer_count-2][i])))
            #if i == 0:
            #    print(sum(local_error)/len(local_error), lambdaa*weight_total)
            w = (sum(local_error)/len(local_error))+lambdaa*weight_total#MSE         old:/2#/len(indiv_highest[0])#NOT SURE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            total_error.append(sum(local_error)/len(local_error))
            #print(output, layer_count)
            sigma = 0
            for o in reversed(output): #other function that gets called for each o, cost and output and network gets past in / the relevant part of the network
                if output.index(o) == 0:
                    #print("first hidden/input " + " " + str(output.index(o)))
                    #print(len(o), 99)
                    
                    for k in range(len(o)):
                        sigmar = cost_net[5][1][k]['layer']*o[k]*(1-o[k])
                        cost_net[5][1][k]["bias"] += sigmar#*w
                        for x in range((neuron_count[0])):
                            cost_net[5][1][k]["weights"][x]+=sigmar*int(img[path][x])+2*lambdaa*network[1][k]["weights"][x]#(before lambda)*w
                        
                    
                elif output.index(o) > layer_count - 2:
                    #print("out of layers " + str(layer_count) + " " + str(output.index(o)))
                    #print(o)
                    #print(o, 0)
                    pass
                elif output.index(o) == layer_count - 2:
                    #print("output_layer " + str(layer_count-2) + " " + str(output.index(o)))
                    #print(neuron_count.index(len(o)), layer_count-1)
                    #print(len(o), 1, len(neuron_count)-neuron_count.index(len(o)), layer_count -1)
                    for s in o:
                        if activation == "sigmoid":
                            if o.index(s) == i:
                                sigma = (s-1)*s*(1-s)
                            else:
                                sigma = (s-0)*s*(1-s)
                        elif activation == "softmax":
                            if o.index(s) == i:
                                sigma = (s-1)
                            else:
                                sigma = (s-0)
                        
                        #print(s, sigma)
                        cost_net[5][layer_count-1][o.index(s)]['bias']+=sigma#*local_error[o.index(s)]#layer_count not -2 but -1 becouse for first hidden/inlayer its 1 and not 0
                        for v in range(len(output[output.index(o)-1])):
                            cost_net[5][layer_count-1][o.index(s)]['weights'][v] += sigma*output[output.index(o)-1][v]#+ 2*lambdaa*network[layer_count-1][o.index(s)]['weights'][v] ##(before lambda)*local_error[o.index(s)]
                            cost_net[5][layer_count-2][v]["layer"] += sigma*network[layer_count-1][o.index(s)]["weights"][v]
                else:
                    #print(len(o), 5)
                    for k in range(len(o)):
                        
                        sigmar = cost_net[5][neuron_count.index(len(o))][k]['layer']*o[k]*(1-o[k])
                        cost_net[5][neuron_count.index(len(o))][k]["bias"] += sigmar*w
                        
                        ö = 0
                        for x in range(len(output[output.index(o)-1])):#not sure about the whole neuron count.index - different numbers
                            #print(x, neuron_count[neuron_count.index(len(o))-1], len(o), neuron_count.index(len(o)))
                            
                            ö +=1
                            cost_net[5][neuron_count.index(len(o))][k]["weights"][x]+=sigmar*output[output.index(o)-1][x]*w
                            
                            cost_net[5][neuron_count.index(len(o))-1][x]["layer"] += sigmar*network[neuron_count.index(len(o))][k]["weights"][x]
                        #print(ö, len(output[output.index(o)-1]), len(cost_net[5][neuron_count.index(len(o))][k]["weights"]))
                    
                    

            #for o in range(len(outlayer_output)):
            #    if o != i:
            #        sigmar = (outlayer_output[o]-0)*outlayer_output[o]*(1-outlayer_output[o]) #poss mean error, mse in local_error and w
            #    else:
            #        sigmar = (outlayer_output[o]-1)*outlayer_output[o]*(1-outlayer_output[o])
            #    cost_net[5][2][o]['bias']+=sigmar*local_error[o]
            #    for v in range(len(layer1_output)):
            #        cost_net[5][2][o]['weights'][v] += sigmar*layer1_output[v]*local_error[o]
            #        cost_net[5][1][v]['layer'] += sigmar*network[2][o]["weights"][v]
            #for k in range(len(layer1_output)):
            #    sigmar = cost_net[5][1][k]['layer']*layer1_output[k]*(1-layer1_output[k])
            #    cost_net[5][1][k]["bias"] += sigmar*w
            #    for x in range((neurons_inlayer)):
            #        cost_net[5][1][k]["weights"][x]+=sigmar*int(img[path][x])*w #sigmar *weightsthen has to be transferred to the next layer
        indiv_highest.append(lool)
        indiv_highest.pop(0)
        #print(indiv_highest[0], a)
        #20bcouse**4#*len(indiv_highest)                  !!!! change to taking the len of previous indiv heigest, and multiplying it uptop !!!!
        #weight updates
        #print(cost_net[5][0])
        for m in network:
            if network.index(m) != 0:
                for g in m:
                    
                    t = network.index(m)
                    z = m.index(g)
                    g["bias"] -= learning_rate*cost_net[5][t][z]["bias"]/10#normal weight update#
                    for y in range(5): #momentum
                        g["bias"] -= learning_rate*((cost_net[4-y][t][z]["bias"]/10)/(y+1.5))
                    for h in range(len(network[t-1])):
                        g["weights"][h]-=learning_rate*cost_net[5][t][z]["weights"][h]/10
                        for b in range(5): #momentum
                            g["weights"][h] -= learning_rate*((cost_net[4-b][t][z]["weights"][h]/10)/(b+1.5))
                    
        #print(cost_net[5][2][0])
        #print(uoliuc)



        if right + wrong == 1000:
            train_acc.append(right/10)
            right = 0
            wrong = 0
        #for g in range(neurons_layer1):
        #    network[1][g]["bias"] -= learning_rate*cost_net[5][1][g]["bias"]/10#normal weight update#
        #    for y in range(5): #momentum
        #        network[1][g]["bias"] -= learning_rate*((cost_net[4-y][1][g]["bias"]/10)/(y+1.5))
        #    for h in range(neurons_inlayer):
        #        network[1][g]["weights"][h]-=learning_rate*cost_net[5][1][g]["weights"][h]/10
        #        for b in range(5):
        #            network[1][g]["weights"][h] -= learning_rate*((cost_net[4-b][1][g]["weights"][h]/10)/(b+1.5))
        #for c in range(neurons_outlayer):
        #    network[2][c]["bias"]-=learning_rate*cost_net[5][2][c]["bias"]/10
        #    for y in range(5):
        #        network[1][c]["bias"] -= learning_rate*((cost_net[4-y][1][c]["bias"]/10)/(y+1.5))
        #    for j in range(neurons_layer1):
        #        network[2][c]["weights"][j]-=learning_rate*cost_net[5][2][c]["weights"][j]/10
        #        for b in range(5):
        #            network[1][c]["weights"][j] -= learning_rate*((cost_net[4-b][1][c]["weights"][j]/10)/(b+1.5))
        total_error_av = (sum(total_error)/len(total_error))
        #if ö % 600 == 0:
        #    print(total_error_av)
        #    print(learning_rate)
        cost_net.pop(0)

    return network

def train_network_fullbatch(network):
    cost_net = load_network("Vcost_net.pkl")
    print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
    new_network = 0
    total_error = []
    for i in range(43):
        for u in range (5):
            for r in range(30):
                if r < 10:
                    path = "gtsrb/" + str(i) + "/0000" + str(u) + "_0000" + str(r) + ".ppm"
                else:
                    path = "gtsrb/" + str(i) + "/0000" + str(u) + "_000" + str(r) + ".ppm"

                layer1_output, outlayer_output, highest = test_network(network, path)
                local_error = []
                for d in range(neurons_layer1):
                    cost_net[1][d]['layer'] = 0
                for p in range(len(outlayer_output)):
                    if p == i:
                        r = (1-outlayer_output[p])
                    else:
                        r = (0-outlayer_output[p])
                    local_error.append(r)
                w = sum(local_error)/len(local_error)#NOT SURE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                total_error.append(w)
                for o in range(len(outlayer_output)):
                    if o != i:
                        sigmar = (outlayer_output[o]-0)*outlayer_output[o]*(1-outlayer_output[o])
                    else:
                        sigmar = (outlayer_output[o]-1)*outlayer_output[o]*(1-outlayer_output[o])
                    cost_net[2][o]['bias']+=sigmar*w
                    for v in range(len(layer1_output)):
                        cost_net[2][o]['weights'][v] += sigmar*layer1_output[v]*w
                        cost_net[1][v]['layer'] += sigmar*network[2][o]["weights"][v]
                for k in range(len(layer1_output)):
                    sigmar = cost_net[1][k]['layer']*layer1_output[k]*(1-layer1_output[k])
                    cost_net[1][k]["bias"] += sigmar*w
                    for x in range((neurons_inlayer)):
                        cost_net[1][k]["weights"][x]+=sigmar*img[path][x]*w

    learning_rate = 0.01/(43*5*30)
    for g in range(neurons_layer1):
        network[1][g]["bias"] -= learning_rate*cost_net[1][g]["bias"]
        for h in range(neurons_inlayer):
            network[1][g]["weights"][h]-=learning_rate*cost_net[1][g]["weights"][h]
    for c in range(neurons_outlayer):
        network[2][c]["bias"]-=learning_rate*cost_net[2][c]["bias"]
        for j in range(neurons_layer1):
            network[2][c]["weights"][j]-=learning_rate*cost_net[2][c]["weights"][j]

    total_error_av = sum(total_error)/len(total_error)
    print(total_error_av)
    print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, psutil.cpu_percent())
    return network
    #print(cost)
    #savepath = str(round(random.uniform(100, 2000))) + "wei.pkl"
    save_weightof("2229wei.pkl", weightof)

def test_accuracy(network):
    
    correct = 0
    incorrect  = 0
    for u in range (10):
        for i in range(890):
        
            
            path = str(u) + "-" + str(i)
            output = test_network(network, path)
            if output[layer_count-1] == u:
                correct += 1
            else:
                incorrect += 1

    print(f"Correct: " + str(correct) + " Incorrect: " + str(incorrect) + " " + str((100*correct)/(8900)))    

def load_all_images(path):
    # csv file name
    filename = path

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

        


    label_counter = [[0] for _ in range(10)]
    rows=numpy.array(rows)
    b = {}
    for i in range(len(rows)):
        label = rows[i][0]

        a = numpy.reshape(rows[i][1:], (-1, 28))


        
        
        np_array = numpy.array(a)

        cr = Image.fromarray(numpy.uint8(np_array))
        #cr[t].show()
        cr = cr.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 9, -1, -1, -1, -1), 1, 0))
        cr = numpy.array(cr)
        pool_size = (2, 2)
        cr = numpy.max(cr.reshape(cr.shape[0] // pool_size[0], pool_size[0], -1, pool_size[1]), axis=(1, 3))
        cr = cr.flatten()
        
        b[label + "-" + str(label_counter[int(label)][0])] = cr/255
        label_counter[int(label)][0] += 1
        
        
    #b = {}
    #for i in range(43):
    #    for u in range (5):
    #        for r in range(30):
    #            if r < 10:
    #                path = "gtsrb/" + str(i) + "/0000" + str(u) + "_0000" + str(r) + ".ppm"
    #            else:
    #                path = "gtsrb/" + str(i) + "/0000" + str(u) + "_000" + str(r) + ".ppm"
    #            b[path] = load_image(path)

                #print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, psutil.cpu_percent())
    return b

def softmax(a, b):
    total = 0
    for i in b:
        total += numpy.exp(i) 
    total = numpy.exp(a)/total
    return total

def test_network(network, path):
    #out_net = copy.deepcopy(network)
    #prob possible to simplify the next 30 lines to 10, but no motivation                         now motivation :)
    total_output = []
    for p in range(layer_count-1):
        layer_output = []
        act_layer_output = []
        for i in range(neuron_count[p+1]):
            summe = 0
            bias = network[p+1][i]['bias']
            for n in range(neuron_count[p]):
                if p == 0:
                    summe += (int(img[path][n])*network[p+1][i]['weights'][n])
                else:
                    summe += (total_output[p - 1][n]*network[p+1][i]["weights"][n])
            
            #layer_output.append(numpy.round(sigmoid(bias + summe), 8))
     
            act_layer_output.append(bias+summe)#numpy.round(sigmoid(bias + summe), 8))
            
        for i in range(neuron_count[p+1]):
            if activation == "sigmoid":
                layer_output.append(numpy.round(sigmoid(act_layer_output[i]), 8))
            elif activation == "softmax":
                if p == layer_count-2:        
                    layer_output.append(softmax(act_layer_output[i], act_layer_output))#numpy.round(sigmoid(bias + summe), 8))
                else:
                    layer_output.append(numpy.round(sigmoid(act_layer_output[i]), 8))
        total_output.append(layer_output)
    total_output.append(total_output[layer_count-2].index(max(total_output[layer_count-2])))
    return(total_output)
    #print("a")    
    #print(total_output[0], total_output[1], total_output[layer_count-2].index(max(total_output[layer_count-2])))
    return [total_output[0], total_output[1], total_output[layer_count-2].index(max(total_output[layer_count-2]))]




    layer1_output = []
    for i in range(neurons_layer1):
        summe = 0
        bias = network[1][i]['bias']
        for n in range(neurons_inlayer):
            
            summe += (int(img[path][n])*network[1][i]['weights'][n])

        layer1_output.append(numpy.round(sigmoid(bias + summe), 5))


    outlayer_output = []
    for i in range(neurons_outlayer):
        summe = 0
        bias = network[2][i]['bias']
        for n in range(neurons_layer1):
            summe += (network[2][i]["weights"][n]*layer1_output[n])

        outlayer_output.append(numpy.round(sigmoid(bias + summe), 5))


    #print(outlayer_output)

    highest = max(outlayer_output)
    for z in range(len(outlayer_output)):
        if outlayer_output[z] == highest:
            print("b")
            print(layer1_output, outlayer_output, z)
            print(hkhkjdfxkjjk)
            return layer1_output, outlayer_output, z

neurons_inlayer = 196
neurons_layer1 = 100
neurons_outlayer = 10
lambdaa = 0.00005
activation = "softmax"
regulisation = "L2"

learning_rate = 1
neuron_count = [196, 100, 10]
layer_count = len(neuron_count)

train_acc = []
#en = create_network()
#wr = create_random_network()
img = load_all_images("mnist_train.csv")
#next picture convolution and softmax
wr = load_network("net/Wand_net_196_100_10_025.pkl")


#save_network("net/Wand_net_196_100_10_025.pkl", tw)
#wc = load_network('8102net.pkl')
#tw = load_network('net/1004net.pkl')#0,015
#wr = load_network('net/1018net.pkl')
#print(tw[1][0]["weights"], tw[2][0]["weights"])
#print(wc)
#rand_wei, rand_net = randomize_network(w, n, -1, 1)
wr = train_network_batch(wr, learning_rate)
#for i in range(2):
#    learning_rate -= learning_rate/1.5
#    tw = train_network_batch(tw, learning_rate)
#    print(i)
save_network("net/1044net.pkl", wr)
#s = str(round(random.uniform(1000000, 2000000)))
#path_wei = "NEW_def_wei.pkl"
#path_net = "NEW_def_net.pkl"
#save_network(path_net, en)
#save_weightof(path_wei, we)
print(train_acc)

#wr = load_network("8007net.pkl")
#train_network(wc, n)
g = test_network(wr, '6-100')
c = test_network(wr, '7-100')
f = test_network(wr, '8-100')
#print(n[2][6]['bias'])
#print(n)
#print(f"this is original: ", g[layer_count-2])
#print(f"this is without cost: ", c[layer_count-2])
#print(f"this is with cost   : ", f[layer_count-2])
#subtracted = []
#for i in range(len(e)):
#    item = e[i] - r[i]
#    subtracted.append(item)

#print(subtracted)
#print(sum(subtracted))
#test_accuracy(wr)
img = load_all_images("mnist_test.csv")
test_accuracy(wr)
#test_accuracy(wc)
#go through 10 pictures, for each pic and weight, if its supposed to be higher add 0.1, if lower take off 0.1

#1 epoch: 2800 lr:0,6/1000 1002
#1 epoch: 3000 lr:1/1000 1003 43 255

#next idea for train network: when high activation highten weight from layer 0 to 1, and look at the other outlayer outputs, not just highest
#1010 1 epoch lr0,6/1000 196/100/10 w moment 83,3%
#1011 1 epoch lr0,6/1000 196/100/10 w/o moment 79,808988%
#1012 (on basis 1010) lr-lr/1,5 3epoch lr0,6/1000 196/100/10 w moment 83,4% so no improvement
#1013 1 epoch lr0,6/1000 196/100/10 w mom w l2 0,005 86,02% 
#1014 1  epoch lr0,6/1000 196/100/10 w mom w l2 hidden 88,7%
#1015 1  epoch lr0,6/1000 196/100/10 w mom w l1 hidden act 85,3%
#1016 1  epoch lr0,6/1000 196/100/10 w mom w l1 hidden abs 90,528%
#1017 1  epoch lr2/1000 196/100/10 w mom w l1 hidden abs 89,15
#1018 1  epoch lr1/1000 196/100/10 w mom w l1 hidden abs 90,7%
#1019 1  epoch lr1/1000 196/100/10 w mom w l2 hidden 90,135%
#1020 1  epoch lr1/1000 196/100/10 w mom w/o l2 hidden abs w/o error inclusion 89,674 
#1021 1  epoch lr1/1000 196/100/10 w mom w l2/2 hidden abs w/o error inclusion 64,5 
#1022 1  epoch lr1/1000 196/100/10 w mom w l2/2 hidden abs w/o error inclusion softmax 88.3 
#1023 1  epoch lr1/1000 196/100/10 w mom w l2 hidden abs w partly error inclusion softmax 88.7 
#1024 1  epoch lr1/1000 196/100/10 w mom softmax 94%
#1025 1  epoch lr1/1000 196/100/10 w mom  89,67
#1026 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,0001 93.843
#1027 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,0001  (only hidden) 94.135
#1028 1  epoch lr1/1000 196/100/10 w/o mom softmax 91,9
#1029 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,0005  (only hidden) 93,4
#1 epoch lr 0,1 88,46
#1031 1  epoch lr0,5 196/100/10 91
#1032 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,0001  (only hidden) 94.135
#1033 1 epoch lr 0,1 88,46
#1034 1  epoch lr0.1 196/100/10 w/o mom softmax 91,9
#1035 1  epoch lr1/1000 196/100/10 w mom softmax 94%
#1036 1  epoch lr0.5/1000 196/100/10 w mom softmax 94%
#1037 1  epoch lr1,5/1000 196/100/10 w mom softmax 94%
#1038 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,00001   94.135
#1039 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,0001 94.135
#1040 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,0005 94.135
#1041 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,001 94.135
#1042 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,0001  (only hidden) 94.135
#1043 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,0005  (only hidden)
#1044 1  epoch lr1/1000 196/100/10 w mom softmax l2 0,00005  (only hidden)