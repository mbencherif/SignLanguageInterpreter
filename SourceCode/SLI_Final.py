import cv2
import numpy as np
import math
import os

ifLibPresent = True
signDetectionCount = 15
signNum = 0

from operator import itemgetter
#import matplotlib.pyplot as plt   
if ifLibPresent:
    import pyttsx
    engine = pyttsx.init()

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('output')

hmin = 0
vmin = 0
smin = 0
hmax = 179
smax = 255
vmax = 255

# orange glove hsv
lower1 = np.array([0,45,0])
upper1 = np.array([75,255,255])

# cyan glove hsv
lower2 = np.array([70,80,0])
upper2 = np.array([100,255,255])

count_defects = 0
counter=1
finalDefects = []
defectPoints = []
radiusValues = []
angleValues = []

calibDone = False
fingersCalibrated = 0
started = False
thumbLength = 0
indexFingerLength = 0
middleFingerLength = 0
ringFingerLength = 0
littleFingerLength = 0

prevSign = 0
signCounter = [0,0,0,0,0,0,0,0,0] 

a = 0
b = 0
centreOfPalm = ([0,0])

matchThreshold = 0.2

Path = "C:\Python27\Gestures\gestures"

def calculateRadius(p,q,r,midIndex):
    #d1 = cv2.magnitude( q[0]-p[0],q[1]-p[1])
    m1 = 0
    m2 = 0
    centreX = 0
    centreY = 0
    mid1 = ((q[0]+p[0])/2.0,(q[1]+p[1])/2.0)
    mid2 = ((q[0]+r[0])/2.0,(q[1]+r[1])/2.0)
    if (q[0]-p[0]) != 0:
        m1 = ((q[1]-p[1])/float(q[0]-p[0]))
    if (q[0]-r[0]) != 0:
        m2 = ((q[1]-r[1])/float(q[0]-r[0]))
    if (q[0]-p[0]) == 0 and (q[0]-r[0]) == 0:
        return
    elif (q[1]-p[1]) == 0 and (q[1]-r[1]) == 0:
        return
    elif (q[0]-p[0]) == 0 and (q[0]-r[0]) != 0:
        centreY = (q[1]+p[1])/2.0
        centreX = ((q[0] + r[0])/2.0) - (((r[1] - q[1])/float(r[0]-q[0]))*(centreY - ((r[1] + q[1])/2.0)))
    elif (q[0]-p[0]) != 0 and (q[0]-r[0]) == 0:
        centreY = (q[1]+r[1])/2.0
        centreX = ((q[0] + p[0])/2.0) - (((q[1] - p[1])/float(q[0]-p[0]))*(centreY - ((p[1] + q[1])/2.0)))
    elif (q[1]-p[1]) == 0 and (q[1]-r[1]) != 0:
        centreX = (q[0]+p[0])/2.0
        centreY = ((q[1] + r[1])/2.0) - (((r[0] - q[0])/float(r[1]-q[1]))*(centreX - ((r[0] + q[0])/2.0)))
    elif (q[1]-p[1]) != 0 and (q[1]-r[1]) == 0:
        centreX = (q[0]+r[0])/2.0
        centreY = ((q[1] + p[1])/2.0) - (((q[0] - p[0])/float(q[1]-p[1]))*(centreX - ((p[0] + q[0])/2.0)))
    elif m1!=0 and (m2-m1):
        inv_m1 = -1/m1
        centreX = (m1*m2*(p[1]-r[1]) + m2*(q[0]+p[0]) - m1*(q[0]+r[0]))/float(2*(m2 - m1))
        centreY = (inv_m1*(centreX - mid1[0]) + mid1[1])
    #else:
    #    centreX = (m1*m2*(p[1]-r[1]) + m2*(q[0]+p[0]) - m1*(q[0]+r[0]))/float(2*(m2 - m1))
    #    centreY = (q[1])

    radius = cv2.magnitude(centreX-p[0],centreY-p[1])
    radiusValues.append([round(radius[0][0],2),midIndex])

def calculateAngle(p,q,r,midIndex):
    modA = round(cv2.magnitude(q[0]-p[0],q[1]-p[1])[0][0],2)
    modB = round(cv2.magnitude(q[0]-r[0],q[1]-r[1])[0][0],2)
    dot_pr = ((q[0]-p[0])*(q[0]-r[0]))+((q[1]-p[1])*(q[1]-r[1]))
    if modA == 0 or modB == 0:
        angleValues.append([0,midIndex])
    else:
        cos_angle = dot_pr/float(modA*modB)
        cos_angle = min(1,max(cos_angle,-1))
        angleValues.append([round(math.degrees(math.acos(cos_angle)),2),midIndex])

def findCentre(p,q,r):
    m1 = 0
    m2 = 0
    centreX = 0
    centreY = 0
    mid1 = ((q[0]+p[0])/2.0,(q[1]+p[1])/2.0)
    mid2 = ((q[0]+r[0])/2.0,(q[1]+r[1])/2.0)
    if (q[0]-p[0]) !=0:
        m1 = ((q[1]-p[1])/float(q[0]-p[0]))
    if (q[0]-r[0]) != 0:
        m2 = ((q[1]-r[1])/float(q[0]-r[0]))
    if (q[0]-p[0]) ==0 and (q[0]-r[0]) == 0:
        return
    elif (q[1]-p[1]) ==0 and (q[1]-r[1]) == 0:
        return
    elif (q[0]-p[0]) ==0 and (q[0]-r[0]) != 0:
        centreY = (q[1]+p[1])/2.0
        centreX = ((q[0] + r[0])/2.0) - (((r[1] - q[1])/float(r[0]-q[0]))*(centreY - ((r[1] + q[1])/2.0)))
    elif (q[0]-p[0]) !=0 and (q[0]-r[0]) == 0:
        centreY = (q[1]+r[1])/2.0
        centreX = ((q[0] + p[0])/2.0) - (((q[1] - p[1])/float(q[0]-p[0]))*(centreY - ((p[1] + q[1])/2.0)))
    elif (q[1]-p[1]) ==0 and (q[1]-r[1]) != 0:
        centreX = (q[0]+p[0])/2.0
        centreY = ((q[1] + r[1])/2.0) - (((r[0] - q[0])/float(r[1]-q[1]))*(centreX - ((r[0] + q[0])/2.0)))
    elif (q[1]-p[1]) !=0 and (q[1]-r[1]) == 0:
        centreX = (q[0]+r[0])/2.0
        centreY = ((q[1] + p[1])/2.0) - (((q[0] - p[0])/float(q[1]-p[1]))*(centreX - ((p[0] + q[0])/2.0)))
    elif m1!=0:
        inv_m1 = -1/m1
        centreX = (m1*m2*(p[1]-r[1]) + m2*(q[0]+p[0]) - m1*(q[0]+r[0]))/float(2*(m2 - m1))
        centreY = (inv_m1*(centreX - mid1[0]) + mid1[1])
    centre = tuple([int(centreX),int(centreY)])
    return centre

def findCosValue(p,q,r):
    modA = round(cv2.magnitude(q[0]-p[0],q[1]-p[1])[0][0],2)
    modB = round(cv2.magnitude(q[0]-r[0],q[1]-r[1])[0][0],2)
    dot_pr = ((q[0]-p[0])*(q[0]-r[0]))+((q[1]-p[1])*(q[1]-r[1]))
    if modA == 0 or modB == 0:
        return 0
    else:
        cos_angle = dot_pr/float(modA*modB)
        cos_angle = min(1,max(cos_angle,-1))
        return cos_angle


while True:
    finalDefects = []
    defectPoints = []
    wristDefects = []
    palmDefects = []
    radiusValues = []
    angleValues = []
    
    lt = []
    radlt = []
    sortedRadiusList = []
    _, frame = cap.read()

    imhsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imhsv, lower1, upper1)
    outp = cv2.bitwise_and(imhsv, imhsv, mask = mask)
    hsv1,hsv2,gray = cv2.split(outp)
    ret,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh1 = cv2.erode(thresh1,None,iterations = 2)
    thresh1 = cv2.dilate(thresh1,None,iterations = 2)
    #cv2.imshow('output',thresh1)
    _,contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    ci = 0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    if ci >= (len(contours)):
        continue
    cnt=contours[ci]
    cnt1=cnt
    #rectval = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(frame.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),-1)
    cv2.drawContours(drawing,[hull],0,(0,0,255),2)
    #cv2.imshow('output1',drawing)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    count_defects = 0
    depth = 0.0
    vv = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if counter == 8:
            #print defects
            #print 'Length'+str(len(defects))
            counter = 9
            cv2.circle(frame,far,10,[0,0,255],-1)
        if d > 3000 and d < 50000:
            if count_defects == 0:
                defectPoints = [[start,end,far,d]]
                finalDefects = [[start,end,far,d]]
            else:
                finalDefects.extend([[start,end,far,d]])
                defectPoints.extend([[start,end,far,d]])
            #cv2.line(frame,start,far,[0,255,255],5)
            #cv2.line(frame,end,far,[0,255,0],5)
            #cv2.circle(frame,end,5,[0,255,0],-1)
            cv2.circle(frame,far,10,[0,0,255],-1)
            #cv2.circle(frame,start,5,[255,0,0],-1)
            count_defects+=1
            counter = 9
        dist = cv2.pointPolygonTest(cnt,far,True)
    if counter == 8:
        print count_defects
    counter = 2

    encCirc = []
    cntr = [[0, 0]]
    for i in range(len(finalDefects)):
        encCirc.append(finalDefects[i][2])

    encCirc1 = np.array(encCirc)

    if (len(encCirc)):
        cntr,rds = cv2.minEnclosingCircle(encCirc1)
        cntr = tuple([int(cntr[0]),int(cntr[1])])
        cv2.circle(frame,cntr,int(rds), [255,255,255],4)

    if (len(defectPoints)) != 0:
        wristDefects = max(defectPoints, key=lambda x: x[2][1])
        palmDefects =  [x for x in defectPoints if x != wristDefects]
    
        for i in range (len(palmDefects)):
            _,_,dep,_ = palmDefects[i]

        finalDefects.remove(wristDefects)
            
    while ((len(finalDefects)) > 2):
        sortedRadiusList = []
        radiusValues = []
        angleValues = []
        lt = []
        radlt = []
        for i in range (len(finalDefects)):
            if (len(finalDefects)) == 3:
                #print 'length ' + str(3)
                calculateRadius(finalDefects[i][2],finalDefects[i+1][2],finalDefects[i+2][2],i+1)
                calculateAngle(finalDefects[i][2],finalDefects[i+1][2],finalDefects[i+2][2],i+1)
                break
            if i+2 >= (len(finalDefects)):
                if i+1 >= (len(finalDefects)):
                    #print 'i+1 > ' + str(len(finalDefects))
                    calculateRadius(finalDefects[i][2],finalDefects[0][2],finalDefects[1][2],0)
                    calculateAngle(finalDefects[i][2],finalDefects[0][2],finalDefects[1][2],0)
                else:
                    #print 'i+2 > ' + str(len(finalDefects))
                    calculateRadius(finalDefects[i][2],finalDefects[i+1][2],finalDefects[0][2],i+1)
                    calculateAngle(finalDefects[i][2],finalDefects[i+1][2],finalDefects[0][2],i+1)
            else:
                #print 'i+2 < ' + str(len(finalDefects))
                calculateRadius(finalDefects[i][2],finalDefects[i+1][2],finalDefects[i+2][2],i+1)
                calculateAngle(finalDefects[i][2],finalDefects[i+1][2],finalDefects[i+2][2],i+1)
        #print 'found radius and angle'
        sortedRadiusList = sorted(radiusValues, key=itemgetter(0), reverse=True)
        for i in range ((len(sortedRadiusList))-1):
            if sortedRadiusList[i][0] == sortedRadiusList[i+1][0]:
                j1 = [sortedRadiusList[i][1],sortedRadiusList[i+1][1]]
                j2 = [x for x in angleValues if x[1] in j1]
                if j2[1][0] > j2[0][0]:
                    sortedRadiusList[i],sortedRadiusList[i+1] = sortedRadiusList[i+1],sortedRadiusList[i]
        #print 'sorted radius list '
        #print sortedRadiusList
        lt = [x for x in angleValues if x[1]==sortedRadiusList[0][1]]
        #print 'found max'
        if lt[0][0] < 90:
            #print 'found centre'
            break
        elif(len(lt))!=0:
            #print 'remove point ' + str(len(lt)) + str(lt[0][1]) + str(len(finalDefects))
            finalDefects.remove(finalDefects[lt[0][1]])

    if (len(finalDefects)) == 2:
        pt1 = finalDefects[0][2]
        pt2= finalDefects[1][2]
        midpt = ((pt1[0]+pt2[0])/2,(pt1[1]+pt2[1])/2)
        radius = cv2.magnitude(midpt[0]-pt1[0],midpt[1]-pt1[1])
        #cv2.circle(frame,midpt,8, [0,0,0],-1)
        #cv2.circle(frame,midpt,int(round(radius[0][0],2)), [0,0,0],2)
    elif (len(finalDefects)) > 2:
        if lt[0][1]+2 >= (len(finalDefects)):
            if lt[0][1]+1 >= (len(finalDefects)):
                centreOfPalm = findCentre(finalDefects[lt[0][1]][2],finalDefects[0][2],finalDefects[1][2])
            else:
                centreOfPalm = findCentre(finalDefects[lt[0][1]][2],finalDefects[lt[0][1]+1][2],finalDefects[0][2])
        else:
            centreOfPalm = findCentre(finalDefects[lt[0][1]][2],finalDefects[lt[0][1]+1][2],finalDefects[lt[0][1]+2][2])
        #cv2.circle(frame,centreOfPalm,8,[0,0,0],-1)
        radlt = [x for x in sortedRadiusList if x[1]==lt[0][1]]
        #cv2.circle(frame,centreOfPalm,int(round(radlt[0][0],2)), [0,0,0],2)
        #print 'centre is' + str(centreOfPalm)

    #print 'end'

    hull = cv2.convexHull(cnt)
    minimum_bb_minXY = hull.min(0)
    minimum_bb_maxXY = hull.max(0)

    #cv2.rectangle(frame, tuple([rectval[0],rectval[1]]), tuple([rectval[0]+rectval[2],rectval[1]+rectval[3]]), [255,0,0],2)

    minrect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(minrect)
    box = np.int0(box)
    cv2.drawContours(frame,[box],0,(255,255,255),3)

    '''d1 = round(cv2.magnitude(box[0][0]-box[1][0],box[0][1]-box[1][1])[0][0],2)
    d2 = round(cv2.magnitude(box[0][0]-box[2][0],box[0][1]-box[2][1])[0][0],2)

    if d1>d2:
        q2 = tuple([(box[0][0]+box[2][0])/2,(box[0][1]+box[2][1])/2])   #centre of shortest side of rectangle
        q1 = tuple([(box[0][0]+q2[0])/2,(box[0][1]+q2[1])/2])           #quarter points q1 q3
        q3 = tuple([(box[2][0]+q2[0])/2,(box[2][1]+q2[1])/2])
        p2 = tuple([(box[1][0]+box[3][0])/2,(box[1][1]+box[3][1])/2])   #centre of the other shortest side of rectangle
        p1 = tuple([(box[1][0]+p2[0])/2,(box[1][1]+p2[1])/2])
        p3 = tuple([(box[3][0]+p2[0])/2,(box[3][1]+p2[1])/2])
        cv2.rectangle(frame, q1, p2, [255,0,0],2)
    rows,cols = thresh1.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(frame,(cols-1,righty),(0,lefty),(255,255,255),2)'''

    #cv2.rectangle(frame, tuple([minimum_bb_minXY[0][0],minimum_bb_minXY[0][1]]), tuple([minimum_bb_maxXY[0][0],minimum_bb_maxXY[0][1]]), [255,0,0],2)

    centreOfMinBBRect = tuple([(minimum_bb_minXY[0][0]+minimum_bb_maxXY[0][0])/2,(minimum_bb_minXY[0][1]+minimum_bb_maxXY[0][1])/2])

    #minimum_bb_minXY = min(box,key=itemgetter(0))
    #minimum_bb_maxXY = max(box,key=itemgetter(0))

    #centreOfMinBBRect = tuple([(minimum_bb_minXY[0]+minimum_bb_maxXY[0])/2,(minimum_bb_minXY[1]+minimum_bb_maxXY[1])/2])

    cv2.circle(frame,centreOfMinBBRect,5,[255,0,255],5)

    #cv2.rectangle(frame, tuple([5,7]), tuple([500,500]), [255,0,0],5)

    group = False
    maxCosValue = 0
    localMaxPoints = []
    fingerTipPoints = []
    mValue = 20
    cosThreshold = 0.6

    cv2.drawContours(frame,[cnt],0,(0,255,0),2)

    for i in range (0,(len(cnt)-mValue)):
        ptX1 = tuple([cnt[i-mValue][0][0],cnt[i-mValue][0][1]])
        ptX = tuple([cnt[i][0][0],cnt[i][0][1]])
        ptX2 = tuple([cnt[i+mValue][0][0],cnt[i+mValue][0][1]])
        #cv2.circle(frame, ptX1, 5, [0,0,0], 5)
        #cv2.circle(frame, ptX, 5, [0,255,0], 5)
        #cv2.circle(frame, ptX2, 5, [0,0,0], 5)
        cosValue = findCosValue(ptX1,ptX,ptX2)
        if (cosValue <= cosThreshold) and (not group):
            group = False
            #cv2.circle(frame, ptX, 5, [0,0,0], 5)
        elif (cosValue > cosThreshold) and (not group):
            group = True
            maxCosValue = cosValue
            #cv2.circle(frame, ptX, 5, [0,0,0], 5)
        elif (cosValue > cosThreshold) and group and (cosValue > maxCosValue):
            maxCosValue = cosValue
            #cv2.circle(frame, ptX, 5, [0,0,0], 5)
        elif (cosValue <= cosThreshold) and group:
            localMaxPoints.append([ptX1,ptX,ptX2])
            #cv2.circle(frame, ptX1, 5, [0,0,0], 5)
            #cv2.circle(frame, ptX, 5, [0,0,0], 5)
            #cv2.circle(frame, ptX2, 5, [0,0,0], 5)
            group = False
        #else:
            #cv2.circle(frame, ptX, 5, [0,0,0], 5)

    for i in range (len(localMaxPoints)):
        cosValue1 = findCosValue(centreOfMinBBRect,localMaxPoints[i][1],localMaxPoints[i][0])
        angleValue1 = round(math.degrees(math.acos(cosValue1)),2)
        cosValue2 = findCosValue(centreOfMinBBRect,localMaxPoints[i][1],localMaxPoints[i][2])
        angleValue2 = round(math.degrees(math.acos(cosValue2)),2)
        if angleValue1 >= 90 and angleValue2 >= 90:
            continue
        elif angleValue1<90 and angleValue2<90:
            fingerTipPoints.append(localMaxPoints[i][1])
            cv2.circle(frame, localMaxPoints[i][1], 5, [0,255,255], 5)

    #print 'len = '
    #print len(fingerTipPoints)
    if (len(fingerTipPoints) == 1):
        for file in os.listdir(Path):
            if file.endswith(".jpg"):
                if(file == '1one.jpg')or(file == '1one1.jpg')or(file == '1one2.jpg') or (file == '1one3.jpg') or (file == '1one4.jpg'):
                    img2 = cv2.imread(Path+'\\'+file)
                else:
                    continue
                imhsv = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(imhsv, lower2, upper2)
                outp1 = cv2.bitwise_and(imhsv, imhsv, mask = mask1)
                hsv1,hsv1,gray1 = cv2.split(outp1)

                ret1,thresh1 = cv2.threshold(gray1,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                thresh1 = cv2.erode(thresh1,None,iterations = 2)
                thresh1 = cv2.dilate(thresh1,None,iterations = 2)
                #cv2.imshow('img2',thresh1)

                _,contours2, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                max_area1 = 0
                ci1 = 0
                for i in range(len(contours2)):
                    cnt2=contours2[i]
                    area1 = cv2.contourArea(cnt2)
                    if(area1>max_area1):
                            max_area1=area1
                ci1=i

                if ci1 >= (len(contours2)):
                    continue

                cnt2=contours2[ci1]

                drawing1 = np.zeros(frame.shape,np.uint8)
                cv2.drawContours(drawing1,[cnt1],0,(0,255,140),-1)
                #cv2.imshow('img1Contour',drawing1)

                drawing2 = np.zeros(frame.shape,np.uint8)
                cv2.drawContours(drawing2,[cnt2],0,(0,255,140),-1)
                #cv2.imshow('img2Contour',drawing2)


                ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
                #print('Matching Result:')
                print(ret)
                if ret <= matchThreshold:
                    print('One')
                    if prevSign == 1:
                        signCounter[0] += 1
                    #else:
                        #signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                    if signCounter[0] == signDetectionCount:
                        signCounter[0] = 0
                        signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                        cv2.putText(frame, "ONE", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
                        if ifLibPresent:
                            engine.say("ONE")
                            engine.runAndWait()

                    prevSign = 1

    elif (len(fingerTipPoints) == 2):
        for file in os.listdir(Path):
            if file.endswith(".jpg"):
                if(file == '2two.jpg')or(file == '2two1.jpg')or(file == '2two2.jpg')or(file == '2two3.jpg')or(file == '2two4.jpg'):
                        img2 = cv2.imread(Path + '\\' + file)
                else:
                        continue
                imhsv = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(imhsv, lower2, upper2)
                outp1 = cv2.bitwise_and(imhsv, imhsv, mask = mask1)

                hsv1,hsv1,gray1 = cv2.split(outp1)

                ret1,thresh1 = cv2.threshold(gray1,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                thresh1 = cv2.erode(thresh1,None,iterations = 2)
                thresh1 = cv2.dilate(thresh1,None,iterations = 2)

                #cv2.imshow('img2',thresh1)

                _,contours2, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                max_area1 = 0
                ci1 = 0
                for i in range(len(contours2)):
                                cnt2=contours2[i]
                                area1 = cv2.contourArea(cnt2)
                                if(area1>max_area1):
                                                max_area1=area1
                                                ci1=i

                if ci1 >= (len(contours2)):
                        continue

                cnt2=contours2[ci1]

                drawing1 = np.zeros(frame.shape,np.uint8)
                cv2.drawContours(drawing1,[cnt1],0,(0,255,140),-1)
                #cv2.imshow('img1Contour',drawing1)

                drawing2 = np.zeros(frame.shape,np.uint8)
                cv2.drawContours(drawing2,[cnt2],0,(0,255,140),-1)
                #cv2.imshow('img2Contour',drawing2)


                ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
                #print('Matching Result:')
                print(ret)
                if ret <= matchThreshold:
                    print('Two')
                    if prevSign == 2:
                       signCounter[1] += 1
                    #else:
                        #signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                    if signCounter[1] == signDetectionCount:
                        signCounter[1] = 0
                        signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                        cv2.putText(frame, "TWO", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
                        if ifLibPresent:
                            engine.say("TWO")
                            engine.runAndWait()

                    prevSign = 2
					
								
    elif (len(fingerTipPoints) == 3):
        for file in os.listdir(Path):
            if file.endswith(".jpg"):
                if(file == '3three.jpg')or(file == '3three1.jpg')or(file == '3three2.jpg')or(file == '3three3.jpg')or(file == '3three4.jpg'):
                    displayString = ""
                    displayString = "THREE"
                    signNum = 3
                elif (file == '6six.jpg')or(file == '6six1.jpg')or(file == '6six2.jpg')or(file == '6six3.jpg')or(file == '6six4.jpg'):
                    displayString = ""
                    displayString = "SIX"
                    signNum = 6
                elif (file == '7seven.jpg')or(file == '7seven1.jpg')or(file == '7seven2.jpg')or(file == '7seven3.jpg')or(file == '7seven4.jpg'):
                    displayString = ""
                    displayString = "SEVEN"
                    signNum = 7
                elif (file == '8eight.jpg')or(file == '8eight1.jpg')or(file == '8eight2.jpg')or(file == '8eight3.jpg')or(file == '8eight4.jpg'):
                    displayString = ""
                    displayString = "EIGHT"
                    signNum = 8
                elif (file == '9nine.jpg')or(file == '9nine1.jpg')or(file == '9nine2.jpg')or(file == '9nine3.jpg')or(file == '9nine4.jpg'):
                    displayString = ""
                    displayString = "NINE"
                    signNum = 9
                else:
                    continue
                img2 = cv2.imread(Path + '\\' + file)
                imhsv = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(imhsv, lower2, upper2)
                outp1 = cv2.bitwise_and(imhsv, imhsv, mask = mask1)

                hsv1,hsv1,gray1 = cv2.split(outp1)

                ret1,thresh1 = cv2.threshold(gray1,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                thresh1 = cv2.erode(thresh1,None,iterations = 2)
                thresh1 = cv2.dilate(thresh1,None,iterations = 2)

                #cv2.imshow('img2',thresh1)

                _,contours2, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                max_area1 = 0
                ci1 = 0
                for i in range(len(contours2)):
                                cnt2=contours2[i]
                                area1 = cv2.contourArea(cnt2)
                                if(area1>max_area1):
                                                max_area1=area1
                                                ci1=i

                if ci1 >= (len(contours2)):
                        continue

                cnt2=contours2[ci1]

                drawing1 = np.zeros(frame.shape,np.uint8)
                cv2.drawContours(drawing1,[cnt1],0,(0,255,140),-1)
                #cv2.imshow('img1Contour',drawing1)

                drawing2 = np.zeros(frame.shape,np.uint8)
                cv2.drawContours(drawing2,[cnt2],0,(0,255,140),-1)
                #cv2.imshow('img2Contour',drawing2)


                ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
                #print('Matching Result:')
                #print(displayString)
                print(ret)
                if ret <= matchThreshold:
                    print(displayString)
                    if prevSign == signNum:
                       signCounter[signNum-1] += 1
                    #else:
                        #signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                    if signCounter[signNum-1] == signDetectionCount:
                        signCounter[signNum-1] = 0
                        signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                        cv2.putText(frame, displayString, (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
                        if ifLibPresent:
                            engine.say(displayString)
                            engine.runAndWait()

                    prevSign = signNum
				
    elif (len(fingerTipPoints) == 4):
        for file in os.listdir(Path):
            if file.endswith(".jpg"):
                if(file == '4four.jpg')or(file == '4four1.jpg')or(file == '4four2.jpg')or(file == '4four3.jpg')or(file == '4four4.jpg'):
                    img2 = cv2.imread(Path + '\\' + file)
                else:
                    continue
                imhsv = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(imhsv, lower2, upper2)
                outp1 = cv2.bitwise_and(imhsv, imhsv, mask = mask1)

                hsv1,hsv1,gray1 = cv2.split(outp1)

                ret1,thresh1 = cv2.threshold(gray1,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                thresh1 = cv2.erode(thresh1,None,iterations = 2)
                thresh1 = cv2.dilate(thresh1,None,iterations = 2)

                #cv2.imshow('img2',thresh1)

                _,contours2, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                max_area1 = 0
                ci1 = 0
                for i in range(len(contours2)):
                                cnt2=contours2[i]
                                area1 = cv2.contourArea(cnt2)
                                if(area1>max_area1):
                                                max_area1=area1
                                                ci1=i

                if ci1 >= (len(contours2)):
                        continue

                cnt2=contours2[ci1]

                drawing1 = np.zeros(frame.shape,np.uint8)
                cv2.drawContours(drawing1,[cnt1],0,(0,255,140),-1)
                #cv2.imshow('img1Contour',drawing1)

                drawing2 = np.zeros(frame.shape,np.uint8)
                cv2.drawContours(drawing2,[cnt2],0,(0,255,140),-1)
                #cv2.imshow('img2Contour',drawing2)


                ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
                #print('Matching Result:')
                print(ret)
                if ret <= matchThreshold:
                    print('four')
                    if prevSign == 4:
                       signCounter[3] += 1
                    #else:
                       #signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                    if signCounter[3] == signDetectionCount:
                        #print('four')
                        signCounter[3] = 0
                        signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                        cv2.putText(frame, "FOUR", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
                        if ifLibPresent:
                            engine.say("FOUR")
                            engine.runAndWait()

                    prevSign = 4

    else:
        for file in os.listdir(Path):
            if file.endswith(".jpg"):
                if (file == '5five.jpg')or(file == '5five1.jpg')or(file == '5five2.jpg')or(file == '5five3.jpg')or(file == '5five4.jpg'):
                    img2 = cv2.imread(Path + '\\' + file)
                else:
                    continue
                imhsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(imhsv, lower2, upper2)
                outp1 = cv2.bitwise_and(imhsv, imhsv, mask=mask1)

                hsv1, hsv1, gray1 = cv2.split(outp1)

                ret1, thresh1 = cv2.threshold(gray1, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh1 = cv2.erode(thresh1, None, iterations=2)
                thresh1 = cv2.dilate(thresh1, None, iterations=2)

                #cv2.imshow('img2', thresh1)

                _, contours2, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                max_area1 = 0
                ci1 = 0
                for i in range(len(contours2)):
                    cnt2 = contours2[i]
                    area1 = cv2.contourArea(cnt2)
                    if (area1 > max_area1):
                        max_area1 = area1
                        ci1 = i

                if ci1 >= (len(contours2)):
                    continue

                cnt2 = contours2[ci1]

                drawing1 = np.zeros(frame.shape, np.uint8)
                cv2.drawContours(drawing1, [cnt1], 0, (0, 255, 140), -1)
                #cv2.imshow('img1Contour', drawing1)

                drawing2 = np.zeros(frame.shape, np.uint8)
                cv2.drawContours(drawing2, [cnt2], 0, (0, 255, 140), -1)
                #cv2.imshow('img2Contour', drawing2)

                ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
                # print('Matching Result:')
                print(ret)
                if ret <= matchThreshold:
                    print('Five')
                    if prevSign == 5:
                       signCounter[4] += 1
                    #else:
                        #signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                    if signCounter[4] == signDetectionCount:
                        signCounter[4] = 0
                        signCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                        cv2.putText(frame, "FIVE", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
                        if ifLibPresent:
                            engine.say("FIVE")
                            engine.runAndWait()

                    prevSign = 5
        
    cv2.imshow('Original',frame)
    cv2.imshow('img2Contour', drawing2)
    cv2.imshow('img1Contour',drawing1)
    
    k = cv2.waitKey(3) & 0xFF
    if k == 27:
        break
    elif k == 115:
        fingersCalibrated += 1
        print int(round(1.5*radlt[0][0],2))
        break

print box
print minimum_bb_minXY
print minimum_bb_maxXY

    
cv2.destroyAllWindows()
cap.release()
