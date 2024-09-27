from collections import Counter
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, Image, ImageFilter
from scipy import ndimage
from skimage import filters,measure
from skimage import measure
from skimage.filters.rank import entropy
from skimage.morphology import disk, dilation, square
from sklearn.cluster import KMeans
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import operator
import math

class imageprocessing:

    def __init__(self, pathconfig):
        self.pathconfig = pathconfig
        # self.plant_classifier_model = keras.models.load_model(pathconfig.plant_classifier_model_path)
        self.yolo_model, self.yolo_classes, self.yolo_colors = self.load_yolo_model()
        self.ValideImageHeight, self.ValideImageWidth = 500, 500
        self.MinAcceptedBlurFactor= 200
        self.FaceDetectionScale=1.1
        self.FaceDetectionMinNeighbors=4
        self.MaxDominantColorPercentageAccepted=70
        self.MinTopColorsCount=6

    def DisplayImage(self, image, title=''):
        plt.title(title)
        plt.imshow(image)
        plt.show()
        
    def DisplayImagesInRow(self, imagelist,columns=2):
        if len(imagelist)>0:
            if len(imagelist)>1:
                while columns>len(imagelist):
                    columns-=1
                
                fig = plt.figure(figsize=(8, 8))
                rows = math.ceil(len(imagelist)/columns)
                imgindex=0
                for i in range(1, columns*rows +1):
                    try:
                        img=imagelist[imgindex]
                        fig.add_subplot(rows, columns, i)
                        plt.imshow(img)
                        imgindex+=1
                    except IndexError:
                        pass
                plt.show()
            else:
                self.DisplayImage(imagelist[0])
        
    def RGB2HEX(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def HEX2RGB(self, color):
        h = color.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def RGB2HSV(self, r, g, b):
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx-mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
        if mx == 0:
            s = 0
        else:
            s = (df/mx)*100
        v = mx*100
        return h, s, v

    def Get_Colors(self, image,number_of_colors):
        modified_image = cv2.resize(image, (200, 200), interpolation = cv2.INTER_AREA)
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

        clf = KMeans(n_clusters = number_of_colors, random_state=100)
        labels = clf.fit_predict(modified_image)
        counts = Counter(labels)

        center_colors = clf.cluster_centers_
        ordered_colors = [center_colors[i] for i in counts.keys()]

        color_dict = dict()
        try:
            for key, value in sorted(counts.items(), key=operator.itemgetter(1), reverse=True):
                color_dict[self.RGB2HEX(ordered_colors[key])]=value
                
            #hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
            #rgb_colors = [ordered_colors[i] for i in counts.keys()]
            hex_colors = color_dict.keys()
            #plt.figure(figsize = (8, 6))
            #plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

            color_list_vals=list(color_dict.values()) 
            sum_color_list_vals=sum(color_list_vals)

            top_1_color_dist = max(color_list_vals)  
            dominantcolorpercent=((top_1_color_dist/sum_color_list_vals)*100)
        
            return dominantcolorpercent,sum_color_list_vals, color_dict
                
        except IndexError:
            return 0, 0 , dict()

    def Get_Color_Difference(self, rgb_colors1, rgb_colors2):
        color1_rgb = sRGBColor(rgb_colors1[0],rgb_colors1[1],rgb_colors1[2]);
        color2_rgb = sRGBColor(rgb_colors2[0],rgb_colors2[1],rgb_colors2[2]);
        color1_lab = convert_color(color1_rgb, LabColor);
        color2_lab = convert_color(color2_rgb, LabColor);
        delta_e = delta_e_cie2000(color1_lab, color2_lab);
        return delta_e
        
    def ExtractNonPlantColorsFromImage(self, img, colortype='nonplant'):
        ## convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)    
        brownColorMask = cv2.inRange(hsv, (10, 100, 20), (30, 255, 200)) #brown color hsv range
        greenColorMask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255)) #green color hsv range
        yellowColorMask = cv2.inRange(hsv, (25, 80, 80), (36, 255, 255)) #yellow color hsv range
        whiteColorMask = cv2.inRange(hsv, (0, 0, 0), (0,0,255)) #white color hsv range
        
        plantcolorMask=greenColorMask
        plantcolorMask = cv2.bitwise_or(plantcolorMask, yellowColorMask)

        if colortype=='nonplant':
            imask = plantcolorMask==0
        else:
            imask = plantcolorMask>0    
        
        colorFilteredImage = np.zeros_like(img, np.uint8)
        colorFilteredImage[imask] = img[imask]
        return colorFilteredImage  

    def UpdateSameColorList(self, matching_colors,hex_inner,hex_outer):
        for idx, same_color_list in enumerate(matching_colors):
            if hex_outer in same_color_list and hex_inner not in same_color_list:
                same_color_list.append(hex_inner)            
                matching_colors[idx]=same_color_list
                break
            if hex_inner in same_color_list and hex_outer not in same_color_list:
                same_color_list.append(hex_outer)
                matching_colors[idx]=same_color_list
                break
                
        return matching_colors

    def Get_Plant_Color_Patterns(self, img,colortype):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        #DisplayImage(img)
        img = self.ExtractNonPlantColorsFromImage(img,colortype)
        #DisplayImage(img)
        domcolorper, sum_color_list_vals, color_dict = self.Get_Colors(img, 10)
        
        if(len(color_dict)==0):
            return 0,0
        
        top_1_color_dist = max(list(color_dict.values()))  
        color_dict = {key:val for key, val in color_dict.items() if val != top_1_color_dist}
        
        mincolorperallowed=2
        color_dict = {key:val for key, val in color_dict.items() if ((val/sum_color_list_vals)*100) > mincolorperallowed}
        
        rgb_colors_array= [self.HEX2RGB(i) for i in color_dict.keys()]
        non_matching_colors=[str(i) for i in color_dict.keys()]
        matching_colors=[]
        
        for rgb_color_outer in rgb_colors_array:
            for rgb_color_inner in rgb_colors_array:
                color_diff = self.Get_Color_Difference(rgb_color_outer, rgb_color_inner)
                hex_outer = self.RGB2HEX(rgb_color_outer)
                hex_inner = self.RGB2HEX(rgb_color_inner)
                
                if int(color_diff)<=30 and int(color_diff)>0:
                    if any(hex_outer in scl for scl in matching_colors) or any(hex_inner in scl for scl in matching_colors):
                        matching_colors = self.UpdateSameColorList(matching_colors,hex_inner,hex_outer)
                    else:
                        same_color_list=[]
                        same_color_list.append(hex_outer)
                        same_color_list.append(hex_inner)
                        matching_colors.append(same_color_list)
                    
                    if hex_outer in non_matching_colors: 
                        non_matching_colors.remove(hex_outer)
                    if hex_inner in non_matching_colors: 
                        non_matching_colors.remove(hex_inner)
                    
        totaluniquecolors=len(matching_colors)+len(non_matching_colors)+1  #1 for less color array 
        return int(domcolorper), totaluniquecolors

    def ColorCoverageAndObjectCount(self, img, colortype): 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        colorFilteredImage = self.ExtractNonPlantColorsFromImage(img, colortype)
        objectcount=0
        colorcoverage=0.0
        
        try:
            gray = cv2.cvtColor(colorFilteredImage, cv2.COLOR_BGR2GRAY)
            otsu_threshold_val = filters.threshold_otsu(gray)
            val=otsu_threshold_val
            leaves = ndimage.binary_fill_holes(gray > val)
            labels = measure.label(leaves, background=1, return_num=False, connectivity=None)
            
            objectcount=labels.max()
            colorcoverage=round(leaves.mean()*100, 3)
        except:
            pass
    
        return colorFilteredImage, colorcoverage,objectcount    

    def EntropyMin(self, image):  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entropy_Of_Image=entropy(gray, disk(20))
        density = np.histogram(entropy_Of_Image, density=True)
        #ent_min_array=min(density,key=lambda item:item[1])
        ent_max_array=max(density,key=lambda item:item[1])
        ent_min=np.amin(ent_max_array)
        ent_max=np.amax(ent_max_array)

        return int(ent_min*1000), int(ent_max*1000)

    def ExtractFeaturesFromImage(self, img):
        original = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        #cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
        matchedcolorcount = 0
        totalobjectscount = 0
        qualifyingsubimages=[]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            perimeter = cv2.arcLength(c,True)
            ROI = original[y:y+h, x:x+w]  
            if perimeter>=20: 
                colorFilteredImage, colorcoverage, objectcount = self.ColorCoverageAndObjectCount(ROI,'plant')
                roiheight, roiwidth, depth = ROI.shape
                totalobjectscount += objectcount                               
                
                if colorcoverage>=15 and objectcount>0 and (roiheight>=50 or roiwidth>=50):
                    #detcount, _, _ = DetectObjects(ROI)
                    notblur = self.Validate_Blur_factor(ROI)
                    entmin, entmax = self.EntropyMin(ROI)
                    entcoeff=(entmax-entmin)/objectcount
                    entcoeff=entcoeff/(roiheight*roiwidth)
                    entcoeff=round(entcoeff*1000,2)
                    ### and entcoeff<=20
                    
                    if notblur is True:    
                        if self.Validate_Plant_Color_Patterns(ROI) is True:
                            roiperemeter=(roiheight*2)+ (roiwidth*2)
                            roiaspectratio = min(roiheight, roiwidth) / max(roiheight, roiwidth)
                            qualifyingsubimages.append((ROI,entcoeff,roiperemeter,roiaspectratio, objectcount))
                    else:
                        matchedcolorcount += 1  
                elif colorcoverage>=15 and objectcount>=1:
                    matchedcolorcount += 1  
                    

        qualifyingsubimages.sort(key = operator.itemgetter(2, 3, 4) , reverse=True)
        topqualifyingsubimages = [i[0] for i in qualifyingsubimages[:10]]
        return len(cnts), matchedcolorcount, totalobjectscount, len(qualifyingsubimages), topqualifyingsubimages

    def ResizeImageToMaxSize(self, img):
        im_pil = Image.fromarray(img)
        maxsize = (2056, 2056)
        im_pil.thumbnail(maxsize, Image.ANTIALIAS)
        cv2img = np.array(im_pil) 
        return cv2img

    def get_best_contours_minmax(self, blurred,original):
        imgheight, imgwidth, imgdepth = original.shape
        imgarea=imgheight*imgwidth
        
        cannyminmaxcombo=[(120,255),(130,255),(5,255),(20,255), (50,160),(100,220),(135,155)]
        cannyminmaxandsizematchcount=[]
        
        sizeproplist=[]
        allcontours=[]
        for eachcomb in cannyminmaxcombo:
            cannymin=eachcomb[0]
            cannymax=eachcomb[1]
            canny = cv2.Canny(blurred, cannymin, cannymax, 1)
            kernel = np.ones((5,5),np.uint8)
            dilate = cv2.dilate(canny, kernel, iterations=1)
            cnts = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
            sizematchcount=0
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                ROI = original[y:y+h, x:x+w]  
                roiheight, roiwidth, depth = ROI.shape
                roiarea=roiheight*roiwidth
                sizeprop=(roiarea/imgarea)*100
                if 0.5<=sizeprop<100: 
                    sizematchcount+=1
                    
            cannyminmaxandsizematchcount.append((cannymin,cannymax,sizematchcount))
        
        cannyminmaxandsizematchcount.sort(key = operator.itemgetter(2) , reverse=True)
        cannymin = cannyminmaxandsizematchcount[:1][0][0]
        cannymax = cannyminmaxandsizematchcount[:1][0][1]

        return cannymin,cannymax

    def get_enhanced(self, img):
        hsvImage= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness_factor=1
        contrast_factor=1
        sharpness_factor=10
        
        image = Image.fromarray(hsvImage)
        enhancer_object = ImageEnhance.Brightness(image)
        image_after_brightness = enhancer_object.enhance(brightness_factor)
        enhancer_object = ImageEnhance.Contrast(image_after_brightness)
        image_after_contrast = enhancer_object.enhance(contrast_factor)
        enhancer_object = ImageEnhance.Sharpness(image_after_contrast)
        image_after_sharpness = enhancer_object.enhance(sharpness_factor)
        
        bgrImg=cv2.cvtColor(np.array(image_after_sharpness), cv2.COLOR_HSV2BGR)  
        
        return bgrImg

    def get_enhanced_2(self, img, brightness_factor=1,contrast_factor=1,sharpness_factor=1):
        pilimage = Image.fromarray(img)
        enhancer_object = ImageEnhance.Brightness(pilimage)
        image_after_brightness = enhancer_object.enhance(brightness_factor)
        enhancer_object = ImageEnhance.Contrast(image_after_brightness)
        image_after_contrast = enhancer_object.enhance(contrast_factor)
        enhancer_object = ImageEnhance.Sharpness(image_after_contrast)
        image_after_sharpness = enhancer_object.enhance(sharpness_factor)
        return np.array(image_after_sharpness)

    def ExtractFeaturesFromImage_New(self, img):
        img = self.ResizeImageToMaxSize(img)
        imgheight, imgwidth, imgdepth = img.shape
        imgarea=imgheight*imgwidth
        original = img.copy()        
        
        gray = cv2.cvtColor(self.get_enhanced(img), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        cannymin, cannymax = self.get_best_contours_minmax(blurred,original)
        canny = cv2.Canny(blurred, cannymin, cannymax, 1)

        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        #cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        qualifyingsubimages=[]
        features_checked_bigsize=0
        features_matched_bigsize=0
        
        features_checked_smallsize=0
        features_matched_smallsize=0
        
        features_checked_microsize=0
        features_matched_microsize=0
        
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            perimeter = cv2.arcLength(c,True)
            ROI = original[y:y+h, x:x+w]  
            roiheight, roiwidth, depth = ROI.shape
            roiarea=roiheight*roiwidth
            sizeprop=(roiarea/imgarea)*100
            
            if 0.5<=sizeprop<100: 
                features_checked_bigsize+=1
                colorFilteredImage, colorcoverage, objectcount = self.ColorCoverageAndObjectCount(ROI,'plant')
                if colorcoverage>=15 and objectcount>0:
                    features_matched_bigsize+=1
                    qualifyingsubimages.append((ROI,sizeprop,perimeter,colorcoverage)) 
                    
            elif sizeprop<0.5 and perimeter>200:
                features_checked_smallsize+=1
                colorFilteredImage, colorcoverage, objectcount = self.ColorCoverageAndObjectCount(ROI,'plant')
                if colorcoverage>=15 and objectcount>0:
                    features_matched_smallsize+=1
                    qualifyingsubimages.append((ROI,sizeprop,perimeter,colorcoverage)) 
                    
            elif perimeter>20:
                features_checked_microsize+=1
                colorFilteredImage, colorcoverage, objectcount = self.ColorCoverageAndObjectCount(ROI,'plant')
                if colorcoverage>=15 and objectcount>0:
                    features_matched_microsize+=1   
                    
        qualifeidfordiagnosis=False
    
        if features_matched_bigsize>=5 or (features_matched_bigsize+features_matched_smallsize)>15 or ((features_matched_bigsize+features_matched_smallsize)>=5 and features_matched_microsize>150) :
            qualifeidfordiagnosis=True
        else:
            qualifeidfordiagnosis=False
        
        qualifyingsubimages.sort(key = operator.itemgetter(1, 2, 3) , reverse=True)
        topqualifyingsubimages = [i[0] for i in qualifyingsubimages[:4]]
        
        return qualifeidfordiagnosis, topqualifyingsubimages

    def load_yolo_model(self):
        # read class names from text file
        classes = None
        with open(self.pathconfig.yolo_classesfile, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes 
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        # read pre-trained model and config file
        net = cv2.dnn.readNet(self.pathconfig.yolo_weightfile, self.pathconfig.yolo_configfile)
        return net, classes, colors

    # function to get the output layer names 
    # in the architecture
    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        unconnlayers= net.getUnconnectedOutLayers()
        if isinstance(unconnlayers[0], np.ndarray):
            output_layers = [layer_names[i[0] - 1] for i in unconnlayers]
        elif isinstance(unconnlayers[0], np.int32):
            output_layers = [layer_names[i - 1] for i in unconnlayers]
        
        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.yolo_classes[class_id])
        color = self.yolo_colors[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    def DetectObjects(self, image):
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392    

        plant_related_items_in_yolo=['banana','apple','orange','broccoli','carrot','potted plant','dining table']
        
        # create input blob 
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        # set input blob for the network
        self.yolo_model.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = self.yolo_model.forward(self.get_output_layers(self.yolo_model))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        objects_detected = dict()
        
        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            if isinstance(i, np.ndarray):
                i = i[0]
            
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            self.draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            
            label = str(self.yolo_classes[class_ids[i]])
            
            if label not in plant_related_items_in_yolo:
                if not label in objects_detected:
                    objects_detected[label] = 1
                else:
                    objects_detected[label] += 1
        
        objlist=[]
        for key, value in objects_detected.items():
            objlist.append(str(value)+' '+key)
        
        return len(objects_detected), ','.join(objlist), cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    def Get_Squared_Image(self, img):
        s = max(img.shape[0:2])
        squared = np.zeros((s,s,3),np.uint8)
        ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2
        squared[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
        return squared

    def Add_Image_Title(self, img, titletext=''):
        title= np.zeros((35, img.shape[1], 3), np.uint8)
        title[:] = (32,32,32) 
        imgwithtitle = cv2.vconcat((title, img))
        # font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX 
        font = cv2.FONT_HERSHEY_COMPLEX 
        fontscale= 1
        textsize = cv2.getTextSize(titletext, font, 1, 2)[0]
        textX = (img.shape[1] - textsize[0]) / 2
        textY = 30
        # cv2.putText(imgwithtitle,titletext,(150,30), font, fontscale,(255,255,255), 1, 0)
        cv2.putText(imgwithtitle, titletext, (int(textX), textY ), font, 1, (255, 255, 255), 2)
        return imgwithtitle

    def Make_Collage(self, featuresimglist, objdetimg):        
        finalmergedimg= None     
        mergedfeaturesimg = None
        bordercolor = [32,32,32]
        resizefactor=(400,400)
        
        if featuresimglist is not None:
            if len(featuresimglist) > 0:
                if len(featuresimglist) > 1:
                    featuresimglist=featuresimglist[:4]

                    bordersformorethan2=[(0, 5, 0, 5),(0, 5, 5, 0),(5, 0, 0, 5),(5, 0, 5, 0)]
                    bordersfor2=[(0, 0, 0, 5),(0, 0, 5, 0)]
                    borders=bordersformorethan2 if len(featuresimglist)>2 else bordersfor2

                    if len(featuresimglist)==3:
                        featuresimglist.append(np.zeros((resizefactor[0],resizefactor[1],3), np.uint8))

                    formattedimglist=[]
                    for idx, img in enumerate(featuresimglist):
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                        #getsqured
                        img= self.Get_Squared_Image(img)
                        #resize
                        img=cv2.resize(img,resizefactor)
                        #addborders
                        img = cv2.copyMakeBorder(img, borders[idx][0], borders[idx][1], borders[idx][2], borders[idx][3], cv2.BORDER_CONSTANT)
                        img=cv2.resize(img,resizefactor)
                        formattedimglist.append(img)

                    if len(featuresimglist)>2:
                        Horizontal1=np.hstack([formattedimglist[0],formattedimglist[1]])
                        Horizontal2=np.hstack([formattedimglist[2],formattedimglist[3]])    
                        Vertical_attachment=np.vstack([Horizontal1,Horizontal2])
                        mergedfeaturesimg = Vertical_attachment
                    
                    elif len(featuresimglist)==2:
                        Vertical_attachment=np.vstack([formattedimglist[0],formattedimglist[1]])
                        mergedfeaturesimg = Vertical_attachment
                else:
                    img = featuresimglist[0] 
                    img= self.Get_Squared_Image(img)
                    img= cv2.resize(img,resizefactor)
                    mergedfeaturesimg = cv2.copyMakeBorder(img, 10,10,10,10, cv2.BORDER_CONSTANT, value=bordercolor)
        
        if objdetimg is not None:
            objdetimg = self.Get_Squared_Image(objdetimg)
            if mergedfeaturesimg is not None:
                imgheight, imgwidth, imgdepth = mergedfeaturesimg.shape
                objdetimg = cv2.resize(objdetimg,(imgheight,imgwidth))
            else:
                objdetimg = cv2.resize(objdetimg,resizefactor)
            
            objdetimg = cv2.copyMakeBorder(objdetimg, 10,10,10,10, cv2.BORDER_CONSTANT, value=bordercolor)
            objdetimg = self.Get_Squared_Image(objdetimg)
            objdetimg = self.Add_Image_Title(objdetimg, 'Detected Items')

        if mergedfeaturesimg is not None:
            if objdetimg is None:
                mergedfeaturesimg = cv2.resize(mergedfeaturesimg,resizefactor)
            mergedfeaturesimg = cv2.copyMakeBorder(mergedfeaturesimg, 10,10,10,10, cv2.BORDER_CONSTANT,value=bordercolor)
            mergedfeaturesimg=self.Get_Squared_Image(mergedfeaturesimg)
            mergedfeaturesimg= self.Add_Image_Title(mergedfeaturesimg, 'Top Features')
        
        if mergedfeaturesimg is not None and objdetimg is not None:  
            Horizontal=np.hstack([objdetimg,mergedfeaturesimg])
            finalmergedimg=Horizontal
        elif mergedfeaturesimg is None and objdetimg is not None:
            finalmergedimg=objdetimg
        elif mergedfeaturesimg is not None and objdetimg is None:
            finalmergedimg=mergedfeaturesimg

        return finalmergedimg                

    def Validate_If_Plant_ByPrediction(self, img):
        try:
            resized=cv2.resize(img,(200,200))
            np_arr=np.array(resized)
            x=np_arr.astype('float32')/255
            #img = keraspreprosseingimage.load_img(test_img_path, target_size=(200, 200))
            #x = keraspreprosseingimage.img_to_array(img)

            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = self.plant_classifier_model.predict(images, batch_size=10)

            predconfidence = format(classes[0][0], 'f')
            predconfidence = float(predconfidence)
            if predconfidence < 0.15:
                return True
            else:
                return False
        except Exception as e:
            print('Error in prediction', e)
            return False

    def Validate_Image_Size(self, img):
        height, width, depth = img.shape
        if height >=self.ValideImageHeight and width>=self.ValideImageWidth:
            return True
        else:
            return False

    def Validate_If_Focussed(self, cv2img):
        image = Image.fromarray(cv2img)
        grey  = image.copy().convert('L')
        edges = grey.filter(ImageFilter.FIND_EDGES)
        selem = square(3)
        fatedges = dilation(np.array(edges),selem)
        #fatedges= get_enhanced_2(np.array(fatedges),1,1,1)   
        
        min_white_thresh=100
        fatedges = cv2.threshold(fatedges, min_white_thresh , 250, cv2.THRESH_BINARY)
        fatedges = fatedges[1]      
        
        # add 1 pixel white border all around
        pad = cv2.copyMakeBorder(fatedges, 1,1,1,1, cv2.BORDER_CONSTANT, value=255)
        h, w = pad.shape
        # create zeros mask 2 pixels larger in each dimension
        mask = np.zeros([h + 2, w + 2], np.uint8)
        # floodfill outer white border with black
        fatedges = cv2.floodFill(pad, mask, (0,0), 0, (5), (0), flags=8)[1]
        # remove border
        fatedges = fatedges[1:h-1, 1:w-1]    

        canny = cv2.Canny(fatedges, 120, 255, 1)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]       
        cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x) , reverse=True)
        # cntsSorted = [i for i in cntsSorted[:10]]
        cntsSorted = [i for i in cntsSorted]
        
        #contrImg = cv2.drawContours(np.array(image), cntsSorted, -1, (255,0,0), 3)
        
        imgheight, imgwidth, imgdepth = np.array(image).shape
        img_area=imgheight*imgwidth
        largest_feature_area=0
        if len(cntsSorted)>0:
            for c in cntsSorted:
                largest_feature_area+=cv2.contourArea(c)
            
        largest_feature_prop= (largest_feature_area/img_area)*100
        
        ispartiallyblur=False
        if largest_feature_prop>=1:
            ispartiallyblur=True
            
        return ispartiallyblur

    def Validate_Blur_factor(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurvalue = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        isfullyblur=False
        ispartiallyblur=False
        if blurvalue < self.MinAcceptedBlurFactor:
            ispartiallyblur= self.Validate_If_Focussed(img)
            if ispartiallyblur==False:
                isfullyblur=True
            
        return isfullyblur, ispartiallyblur

    def Validate_Face_Detection(self, img):
        face_cascade = cv2.CascadeClassifier(self.pathconfig.haat_caascade_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, self.FaceDetectionScale, self.FaceDetectionMinNeighbors, minSize=(50, 50), flags = cv2.CASCADE_SCALE_IMAGE)
        if len(faces)== 0:
            return True
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # cv2.imwrite(os.path.join(TESTDATADIR,'facedetected.jpg'),img)        
            return False      
        
    def Validate_Plant_Color_Patterns(self, img):
        domcolor1, uncolor1 = self.Get_Plant_Color_Patterns(img,'plant')
        domcolor2, uncolor2 = self.Get_Plant_Color_Patterns(img,'nonplant')
        
        if domcolor2==0 or uncolor2==0:
            return False
        
        dom_color_coeff = round((domcolor1/domcolor2), 2)
        un_color_coeff = round((uncolor1/uncolor2), 2)
        
        dom_thresh=1.3
        un_thresh=0.4
        
        if dom_color_coeff<= dom_thresh and un_color_coeff >=un_thresh:
            return True
        elif dom_color_coeff> dom_thresh and un_color_coeff <un_thresh:
            return False
        else:
            return False
        
    def Validate_Plant_Features(self, img):
        totalfeatures, matchedcolorcount, totalobjectscount, subimagescount, topsubimages = self.ExtractFeaturesFromImage(img)
        if totalfeatures>=250 and matchedcolorcount>20:    
            return True,topsubimages
        else:
            return False,topsubimages
        
class image_validator:             
    
    def __init__(self, pathconfig):
        self.img_proc = imageprocessing(pathconfig)

    def Validate_Image(self, img):        
        isvalid=False

        imagequalifiedfordiagnosis = self.img_proc.Validate_If_Plant_ByPrediction(img)
        sizevalid = self.img_proc.Validate_Image_Size(img)
        isfullyblur, ispartiallyblur = self.img_proc.Validate_Blur_factor(img)
        featuresqualifeidfordiagnosis, topsubimages = self.img_proc.ExtractFeaturesFromImage_New(img)
        objdetcount, objdetlist, _ = self.img_proc.DetectObjects(img)

        responses = []
        objdettext=''
        resulttext=''

        if sizevalid is False:
            responses.append('imagesizesmall')
        
        if isfullyblur is True:
            responses.append('imagefullyblur')
        elif ispartiallyblur is True:            
            responses.append('imagepartiallyblur')
        
        if imagequalifiedfordiagnosis and featuresqualifeidfordiagnosis is False:
            responses.append('imagenofeatures')
            
        if objdetcount> 0:
            objdettext = 'objectsdetected'+ ' => {}'
            objdettext = objdettext.format(objdetlist)
            responses.append(objdettext.lower())

        if imagequalifiedfordiagnosis and featuresqualifeidfordiagnosis and sizevalid and isfullyblur is False and objdetcount==0:
            isvalid=True
            resulttext='imagefullyvalid' + '{}{}'
            butstr = ', but ' if ispartiallyblur is True else ''
            resulttext = resulttext.format(butstr, ' and '.join(responses).lower())
        elif imagequalifiedfordiagnosis is False and featuresqualifeidfordiagnosis and len(topsubimages)>0 and isfullyblur is False:
            isvalid=True
            resulttext = 'imagepartiallyvalid' + '{}{}'
            butstr = ', but ' if (objdetcount>0 or ispartiallyblur is True) else ''
            resulttext = resulttext.format(butstr, ' and '.join(responses).lower())
        else:
            resulttext = 'imagenotvalid' + '. {}'
            resulttext = resulttext.format(' and '.join(responses).capitalize())
        
        return isvalid, resulttext


