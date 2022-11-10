from mpi4py import MPI
import numpy as np
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Sequential

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

comm.Barrier()

if rank == 0:
    start = time.time()

def fn_modelfit(model):
    gw = []
    gw1 = []

    for layer in model.layers:
        if rank == 0:
            if not (layer.name == 'flatten' or ('max_pooling') in layer.name):
                print("Results from Master")
                print(layer.name)

                lw1 = comm.recv(source=1, tag=10)
                lw2 = comm.recv(source=2, tag=10)
                lw3 = comm.recv(source=3, tag=10)
                lw4 = comm.recv(source=4, tag=10)
                
                gw1 = lw1+lw2+lw3+lw4
                gw1 = gw1/4
                
                gw1 = comm.bcast(gw1, root=0)
                
                gw.append(gw1)
                
                comm.Barrier()
        else:
            if not (layer.name == 'flatten' or ('max_pooling') in layer.name):

                lw = layer.get_weights()[0]
                
                comm.send(lw, dest=0, tag=10)
                
                gw1 = comm.bcast(gw1,root=0)
                
                np_gw1 = np.array(gw1)
                
                layer.set_weights(
                    [np_gw1, np.ones(layer.get_weights()[1].shape)])
                
                comm.Barrier()

    if rank == 0:
        return gw
    else:
        return None

def exec(dt):

    train_size = int(len(dt)*.7)
    val_size = int(len(dt)*.2)
    test_size = int(len(dt)*.1)+1

    train = dt.take(train_size)
    val = dt.skip(train_size).take(val_size)
    test = dt.skip(train_size+val_size).take(test_size)

    model = Sequential()

    #Add convolution and max-pooling layer
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    model.summary()

    Iterations = 5
    
    for i in range(Iterations):
        history = model.fit(train, epochs=1, validation_data=val)
        weights = fn_modelfit(model)


if rank==0:
    d1 = tf.keras.utils.image_dataset_from_directory("data/data1")
    print("master working")
    print(d1)
    print(len(d1))
    exec(d1)

elif rank == 1:
    print("Worker1 working")
    d2 = tf.keras.utils.image_dataset_from_directory("data/data2")
    print(d2)
    print(len(d2))
    exec(d2)

elif rank == 2:
    print("Worker2 working")
    d3 = tf.keras.utils.image_dataset_from_directory("data/data3")
    print(d3)
    print(len(d3))
    exec(d3)

elif rank == 3:
    print("Worker3 working")
    d4 = tf.keras.utils.image_dataset_from_directory("data/data4")
    print(d4)
    print(len(d4))
    exec(d4)

elif rank == 4:
    print("Worker4 working")
    d1 = tf.keras.utils.image_dataset_from_directory("data/data1")
    print(d1)
    print(len(d1))
    exec(d1)

comm.Barrier()

if rank == 0:
    end = time.time()

    print("Total Time Taken : "+str(end-start))