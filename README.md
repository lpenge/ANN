




# ANN

Damit dieser Code funktioniert, muss dieser Code in VSCode oder ein ähnliches Programm als .py Datei eingefügt werden. Zusüätzlich müssen der trainings- und Testsatz der MNIST-Datenbank als csv Dateien heruntergeladen werden. 
In den letzten 25 Zeilen des codes kann man ändern, wie das Training abläuft (um L2 Regularisierung an / auszuschalten muss in der Funktion "Train_network_batch()" im Gradienten-berechnungsabschnitt die Addiewwrung von Lambda durch # an oder ausgeschalten werden. Der code wird dann mit mnist.py im Terminal ausgeführt, und gibt am Ende die Genauigkeit des Netzwerkes bei dem Erlernen der MNIST-Datenbank.



from PIL import Image, ImageFilter
import copy
import psutil
import pickle
import random
import math
import csv
import numpy

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
    return list(img.getdata())


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

            w = (sum(local_error)/len(local_error))+lambdaa*weight_total#MSE       
            total_error.append(sum(local_error)/len(local_error))
            sigma = 0
            for o in reversed(output): #other function that gets called for each o, cost and output and network gets past in / the relevant part of the network
                if output.index(o) == 0:
                    
                    
                    for k in range(len(o)):
                        sigmar = cost_net[5][1][k]['layer']*o[k]*(1-o[k])
                        cost_net[5][1][k]["bias"] += sigmar
                        for x in range((neuron_count[0])):
                            cost_net[5][1][k]["weights"][x]+=sigmar*int(img[path][x])+2*lambdaa*network[1][k]["weights"][x]
                        
                    
                elif output.index(o) > layer_count - 2:
                    pass
                elif output.index(o) == layer_count - 2:
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
                        cost_net[5][layer_count-1][o.index(s)]['bias']+=sigma
                        for v in range(len(output[output.index(o)-1])):
                            cost_net[5][layer_count-1][o.index(s)]['weights'][v] += sigma*output[output.index(o)-1][v] + 2*lambdaa*network[layer_count-1][o.index(s)]['weights'][v]
                            cost_net[5][layer_count-2][v]["layer"] += sigma*network[layer_count-1][o.index(s)]["weights"][v]
                else:
                    
                    for k in range(len(o)):
                        
                        sigmar = cost_net[5][neuron_count.index(len(o))][k]['layer']*o[k]*(1-o[k])
                        cost_net[5][neuron_count.index(len(o))][k]["bias"] += sigmar*w
                        
                        ö = 0
                        for x in range(len(output[output.index(o)-1])):#not sure about the whole neuron count.index - different numbers
                           
                            
                            ö +=1
                            cost_net[5][neuron_count.index(len(o))][k]["weights"][x]+=sigmar*output[output.index(o)-1][x]*w
                            
                            cost_net[5][neuron_count.index(len(o))-1][x]["layer"] += sigmar*network[neuron_count.index(len(o))][k]["weights"][x]
                    
        indiv_highest.append(lool)
        indiv_highest.pop(0)
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



        if right + wrong == 1000:
            train_acc.append(right/10)
            right = 0
            wrong = 0

        total_error_av = (sum(total_error)/len(total_error))

        cost_net.pop(0)

    return network


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
    return b

def softmax(a, b):

    total = 0
    for i in b:
        total += numpy.exp(i) 
    total = numpy.exp(a)/total
    return total

def test_network(network, path):

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

img = load_all_images("mnist_train.csv")

wr = create_random_network()

wr = train_network_batch(wr, learning_rate)

save_network("net/1044net.pkl", wr)

img = load_all_images("mnist_test.csv")

test_accuracy(wr)
