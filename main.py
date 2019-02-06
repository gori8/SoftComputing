import os
import sys
import numpy as np
import cv2
from keras.models import load_model
from neural_net import trainNN, process
from vector import distance, pnt2line
from scipy import ndimage


videos = ['C:/soft/SoftComputing/data/video-0.avi', 'C:/soft/SoftComputing/data/video-1.avi', 'C:/soft/SoftComputing/data/video-2.avi',
             'C:/soft/SoftComputing/data/video-3.avi', 'C:/soft/SoftComputing/data/video-4.avi', 'C:/soft/SoftComputing/data/video-5.avi', 
             'C:/soft/SoftComputing/data/video-6.avi', 'C:/soft/SoftComputing/data/video-7.avi', 'C:/soft/SoftComputing/data/video-8.avi', 
             'C:/soft/SoftComputing/data/video-9.avi']
    

def number_nn_recogn(img, center):
    
    global model
    x, y = center

    global counter

    croppedImage = img[y-12:y+12,x-12:x+12]

    croppedImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
    
    if croppedImage is not None:
        kernel=(4,4)
        croppedImage = cv2.dilate(croppedImage, kernel)
        croppedImage = process(croppedImage)
        flatImg=croppedImage.flatten()
        nnInput =  flatImg/ 255.0
        nnInput = (np.array([nnInput], 'float32'))
        recogn_number = np.argmax(model.predict(nnInput)[0])
        #cv2.imshow('Cropped Image', croppedImage)
    else:
        recogn_number=-1
    return recogn_number


def in_range(r, ele, objects):
    ret = []
    for o in objects:
        d = distance(ele['center'], o['center'])
        if d<r:
            ret.append(o)
    return ret


def object_tracking(img, linesBG):
    origImg = img.copy()
    global model
    global elements
    global frameNumber
    global counter
    global added
    global subbed
    global times
    global subbedNumbers
    global addedNumbers
    global flag
	
    kernel = np.ones((2, 2),np.uint8)
    lower = np.array([230, 230, 230])
    upper = np.array([255, 255, 255])

    lineAdd = linesBG['add']
    lineSub = linesBG['sub']
    
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    mask = cv2.inRange(img, lower, upper)    
    img0 = 1.0*mask

    img0 = cv2.dilate(img0,kernel)
    img0 = cv2.dilate(img0,kernel)

    labeled, nr_objects = ndimage.label(img0)
    objects = ndimage.find_objects(labeled)
    for i in range(nr_objects):
        loc = objects[i]
        
        (xc,yc) = ((loc[1].stop + loc[1].start)/2,
                   (loc[0].stop + loc[0].start)/2)
        (dxc,dyc) = ((loc[1].stop - loc[1].start),
                   (loc[0].stop - loc[0].start))

        if(dxc>10 or dyc>10):
            dxc = int(dxc)
            dyc = int(dyc)
            xc = int(xc)
            yc = int(yc)
            cv2.circle(img, (xc,yc), 16, (25, 25, 255), 1)
            el = {'center':(xc,yc), 'size':(dxc,dyc), 'frameNumber':frameNumber}
            lst = in_range(20, el, elements)
            nn = len(lst)
            if nn == 0:
                number=number_nn_recogn(origImg, (xc, yc))
                el['number'] = number
                el['frameNumber'] = frameNumber
                el['passedPlus'] = False
                el['passedMinus'] = False
                el['history'] = [{'center':(xc,yc), 'size':(dxc,dyc), 'frameNumber':frameNumber, 'number':number}]
                elements.append(el)
            elif nn == 1:
                number=number_nn_recogn(origImg, (xc, yc))
                lst[0]['center'] = el['center']
                lst[0]['frameNumber'] = frameNumber
                lst[0]['history'].append({'center':(xc,yc), 'size':(dxc,dyc), 'frameNumber':frameNumber, 'number':number}) 
            else:
                for j in range(nn):
                    lst[j]['number']=-1
                    el['number']=-1
            
                        
    for el in elements:
        frameDiff = frameNumber - el['frameNumber']
        if(frameDiff<3):
            history = [a['number'] for a in el['history'] if a['number']!=-1]
            v=np.bincount(history)
            if v.size!=0:
                val = np.argmax(v)
            else:
                val=0
            dist, pnt, r = pnt2line(el['center'], lineAdd[0], lineAdd[1])
            c = None
            passed = False
            if r>0:
                passed = True
                c = (25, 25, 255)
                if(dist<14):
                    c = (0, 255, 160)
                    if el['passedPlus'] == False:
                        added += val
                        addedNumbers.append(val)
                        el['passedPlus'] = True
                        counter += 1
                        
            if passed:
                cv2.circle(img, el['center'], 16, c, 2)
            
            dist, pnt, r = pnt2line(el['center'],tuple(np.add(lineAdd[0],(20,20))), tuple(np.add(lineAdd[1],(20,20))))
            c = None
            passed = False
            if r>0:
                passed = True
                c = (25, 25, 255)
                if(dist<14):
                    c = (0, 255, 160)
                    if el['passedPlus'] == False:
                        added += val
                        addedNumbers.append(val)
                        el['passedPlus'] = True
                        counter += 1
                        

            
            passed = False
            dist, pnt, r = pnt2line(el['center'], lineSub[0], lineSub[1])
            if r>0: #add or dist 
                passed = True
                c = (25, 25, 255)
                if(dist<14):
                    c = (0, 255, 160)
                    if el['passedMinus'] == False: #and not(distance(led['center'],el['center'])<20 and led['number']==el['number']):
                        el['passedMinus'] = True
                        counter += 1
                        subbed -= val
                        subbedNumbers.append(val)
                    
                        

            if passed:
                cv2.circle(img, el['center'], 16, c, 2)
            
            dist, pnt, r = pnt2line(el['center'],tuple(np.add(lineSub[0],(20,30))), tuple(np.add(lineSub[1],(20,30))))
            c = None
            passed = False
            if r>0:
                passed = True
                c = (25, 25, 255)
                if(dist<14):
                    c = (0, 255, 160)
                    if el['passedMinus'] == False:
                        el['passedMinus'] = True
                        counter += 1
                        subbed -= val
                        subbedNumbers.append(val)
            
            
           
            
            if el['number'] is not None:
                history = [a['number'] for a in el['history'] if a['number']!=-1]
                v=np.bincount(history)
                if v.size!=0:
                    val = np.argmax(v)
                else:
                    val=0   
                cv2.putText(img, text = str(val), 
                org = (el['center'][0]+15, el['center'][1]+20),fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = (0, 0, 255))
               
            

    
    cv2.putText(img, text = 'Add: ' + str(added), org = (480, 40), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,255))
    cv2.putText(img, text = 'Sub: ' + str(subbed), org = (480, 60), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,255))    
    cv2.putText(img, text = 'Sum: ' + str(added + subbed), org = (480, 80), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,255))    
    cv2.putText(img, text = 'Counter: ' + str(counter), org = (480, 100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,255))    

    
    

global kord
global minX
global minY
global maxX
global maxY

def find_line():
    global minY
    global minX
    global maxX
    global maxY
    minY=2000
    minX=2000
    maxX=0
    maxY=0
    for c in kord:
        if c[0]<minX:
            minX=c[0]
        if c[1]>maxY:
            maxY=c[1]
        if c[2]>maxX:
            maxX=c[2]
        if c[3]<minY:
            minY=c[3]

def color_filter(img, r, g, b):
    colors = [b, g, r]
    result = np.zeros(img.shape, dtype=np.uint8)
    for i in range(3):
        result[:, :, i] = np.where(img[:, :, i] < colors[i], 0, 255)
    return result.astype(np.uint8)


def detectLines(frame):
    res=color_filter(frame,0,0,200)
    processed_frame = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.Canny(processed_frame,threshold1=20,threshold2=100)
    kernel = np.ones((3,3))
    processed_frame=cv2.erode(cv2.dilate(processed_frame, kernel, iterations=2),kernel,iterations=2)
    lines=cv2.HoughLinesP(processed_frame,1,np.pi/180,200,200,20)
    
    global kord
    kord=[]
    global minY
    global minX
    global maxX
    global maxY
    
    for l in lines:
        coords=l[0]
        kord.append(coords)
    
    find_line()
    linesFinal={}
    linesFinal['add']=((minX ,maxY),(maxX ,minY))
    
    res=color_filter(frame,0,200,0)
    processed_frame = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.Canny(processed_frame,threshold1=20,threshold2=100)
    kernel = np.ones((3,3))
    processed_frame=cv2.erode(cv2.dilate(processed_frame, kernel, iterations=2),kernel,iterations=2)
    lines=cv2.HoughLinesP(processed_frame,1,np.pi/180,200,200,20)
    
    kord=[]
    
    for l in lines:
        coords=l[0]
        kord.append(coords)
    
    find_line()
    
    linesFinal['sub']=((minX ,maxY),(maxX ,minY)) 
    
    
    return linesFinal
     
global nVideo
nVideo=0
global retSum
retSum=None

def main(vid,view):
    
    global retSum
    global elements
    global frameNumber
    global counter
    
    global added
    global addedNumbers
    
    global subbed
    global subbedNumbers
    
    global nVideo
    
    if retSum is None:
        retSum=[]
    
    added = 0
    subbed = 0
    counter=0
    subbedNumbers = []
    addedNumbers = []
    frameNumber = 0
    elements = []
	
    print('\n----------------')

    if view is None:
        view = True

    cap = cv2.VideoCapture(videos[vid])
    
    lines = None

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            frameNumber += 1
            if lines is None:
                lines = detectLines(frame)
            addLine = lines['add']
            subLine = lines['sub']
            object_tracking(frame, lines)
            
            key=cv2.waitKey(1) & 0xFF
            
            if view is True:
                cv2.imshow(videos[vid], frame)
          
            if key == ord('x'):
                break
        else:
            break

    cap.release()

    cv2.destroyAllWindows()


    sum = added + subbed

    
    print('\nResult = ' + str(sum) +', Addition = ' + str(added) + ', Subtraction = ' + str(subbed) + ', Counter = ' + str(counter))
    print('Finished ' + videos[vid])
    print('\n')
    
    
    added = 0
    subbed = 0
    counter=0
    subbedNumbers = []
    addedNumbers = []
    frameNumber = 0
    elements = []
    times=[]
	
    nVideo+=1
    
    retSum.append(sum)
    if nVideo==video_count:
        return retSum



###### clear console ######
os.system('cls')


global model
model = load_model('C:/soft/SoftComputing/model.h5')

global video_count

if int(sys.argv[1]) >= 0 and int(sys.argv[1]) <= 9:
    video_count=1
    vid=int(sys.argv[1])
    main(vid,view = True)
elif int(sys.argv[1]) == 888:
    ###inicijalizuj fajl za upis###
    f = open('C:/soft/SoftComputing/out.txt', 'w')
    strWr = 'RA 23/2015 Igor Antolovic\n'
    strWr += 'file\tsum\n'
    
    video_count=10
    results = []
    for i in range(0, 10):
        results = main(i,view = False)

    for i in range(0, 10):
        strWr += videos[i] + ' ' + str(results[i]) + '\n'

    f.write(strWr)
    f.close()

    print(strWr)
 

    

