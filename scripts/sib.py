import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from scipy import ndimage
from skimage import filters
from skimage import measure
# from scripts.imageprocessing import image_validator
from ultralytics import YOLO
import cv2
import numpy as np
import base64

class SIB:

    def __init__(self, pathconfig):
        self.pathconfig = pathconfig
        self.ultralytics_sib_local_model =  YOLO(self.pathconfig.ultralytics_sib_local_model_path)

    def DetectProblemsInImage(self, base64_str):
        image = self._base64_to_cv2_image(base64_str)

        # validator = image_validator(self.pathconfig)
        # isvalid, resulttext = validator.Validate_Image(image)
        isvalid = True
        podborer_algo_result = None
        tikkaleaf_algo_result = None
        sib_model_detection_results = None
        if isvalid:
            # podborer_algo_result = self.FindGramPodBorerInGroundNut(image)
            # tikkaleaf_algo_result = self.FindTikkaLeafSpotInGroundNut(image)
            sib_model_detection_results = self._detect_from_image(image)

        return podborer_algo_result, tikkaleaf_algo_result, sib_model_detection_results
        

    def FindTikkaLeafSpotInGroundNut(self, image):
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        # Find contours
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Iterate thorugh contours and filter for ROI
        brown_spot_match_count = 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            (cir_x,cir_y),radius = cv2.minEnclosingCircle(c)
            center = (int(cir_x),int(cir_y))
            radius = int(radius)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c,True)

            if radius<=50 and area>=150: 
                ROI = original[y:y+h, x:x+w]

                try:
                    combinedspotcount, combinedcolorcoverage, browncolorcoverage, greencolorcoverage, yellowcolorcoverage = self._count_of_tikka_leaf_spots_and_color_coverage(ROI, False)
                    if combinedcolorcoverage>=90 and browncolorcoverage>=13 and (greencolorcoverage+yellowcolorcoverage) > 40:
                        cv2.circle(image,center,radius,(0,0,255),10)
                        brown_spot_match_count += 1 
                except:
                    i=0

        return self._cv2_image_to_base64(image)

    def FindGramPodBorerInGroundNut(self, image):
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        # Find contours
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Iterate thorugh contours and filter for ROI
        small_circular_patch_count = 0
        large_circular_patch_count = 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            (cir_x,cir_y),cir_radius = cv2.minEnclosingCircle(c)
            cir_center = (int(cir_x),int(cir_y))
            cir_radius = int(cir_radius)
            cir_area=3.14*cir_radius*cir_radius
            cir_perimeter=2*3.14*cir_radius
            
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c,True) 

            if cir_radius<=200 and area>200: 
                ROI = original[y:y+h, x:x+w]
                
                try:
                    circularpatchshapefraction= round(((area/cir_area)/3.14) , 2) #threshold=0.17
                    circularpatchcoverage= round((area/perimeter)*circularpatchshapefraction,2) #threshold=0.70

                    _, greencolorcoverage, objectcount = self._count_and_color_coverage_helper(ROI,"green")
                    if (greencolorcoverage>30 
                        and (circularpatchcoverage>=0.70 and circularpatchcoverage<=2) 
                        and self._check_if_hole(ROI)==True):          
                        if perimeter>=100:
                            large_circular_patch_count+=1
                            cv2.circle(image,cir_center,cir_radius,(0,0,255),10)
                            hull = cv2.convexHull(c)
                            #cv2.drawContours(image, [hull], -10, (102,51,0), 5)   
                        else:
                            small_circular_patch_count += 1 
                            cv2.circle(image,cir_center,cir_radius,(0,0,255),10)
                            hull = cv2.convexHull(c)
                            #cv2.drawContours(image, [hull], -10,(0,0,255), 5) 
                        
                        self._add_text_on_image(image, x,y,str(round(perimeter,2)))
                except:
                    break
                                        
        return self._cv2_image_to_base64(image)
    
    def FindBacterialLeafBlightInPaddy(self, image):
        image_height, image_width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        # Find contours
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Iterate thorugh contours and filter for ROI
        yellow_margin_count = 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            
            mar_rect = cv2.minAreaRect(c)
            mar_box = cv2.boxPoints(mar_rect)
            mar_box = np.int0(mar_box)
            mar_center = mar_rect[0]
            mar_rotation = mar_rect[2]
            mar_width = mar_rect[1][0]
            mar_height = mar_rect[1][1]
            mar_area=mar_width*mar_height
            mar_angle = mar_rect[-1]
            if mar_angle < -45:
                mar_angle = -(90 + mar_angle)
            else:
                mar_angle = -mar_angle
                
            cont_area = cv2.contourArea(c)
            cont_perimeter = cv2.arcLength(c,True) 
            
            blightfraction=(cont_area/mar_area)*cont_perimeter
            size_proportion=(max(mar_height,mar_width))/(max(image_height,image_width))
            size_proportion=size_proportion*100
            
            #if True: 
            if size_proportion>10 and blightfraction>400: 
                yellow_margin_count+=1
                ROI = image[y:y+h, x:x+w]
                hull = cv2.convexHull(c)
                cv2.drawContours(image, [hull], -10,(0,0,255), 12)
                
        return self._cv2_image_to_base64(image)
        
    def _image_to_hsv(self, image):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        return hsvImage

    def _enhance_image(self, hsvImage, brightness_factor,contrast_factor,sharpness_factor):
        #image = Image.open(hsvImage)
        image = Image.fromarray(hsvImage)
        
        enhancer_object = ImageEnhance.Brightness(image)
        image_after_brightness = enhancer_object.enhance(brightness_factor)
        
        enhancer_object = ImageEnhance.Contrast(image_after_brightness)
        image_after_contrast = enhancer_object.enhance(contrast_factor)

        enhancer_object = ImageEnhance.Sharpness(image_after_contrast)
        image_after_sharpness = enhancer_object.enhance(sharpness_factor)
        
        return image_after_sharpness

    def _get_enhanced(self, img):
        hsvImage= self._image_to_hsv(img)
        #enhanced_image = _enhance_image(hsvImage, brightness_factor=1.2,contrast_factor=2,sharpness_factor=10)
        #enhanced_image = _enhance_image(hsvImage, brightness_factor=1.2,contrast_factor=2,sharpness_factor=20)
        enhanced_image = self._enhance_image(hsvImage, brightness_factor=1,contrast_factor=1,sharpness_factor=10)
        bgrImg=cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_HSV2RGB)    
        return bgrImg

    def _get_exact_contour(self, conImg):
        gray = cv2.cvtColor(conImg, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)
        cnt, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        hull = cv2.convexHull(cnt[0])
        x,y,w,h = cv2.boundingRect(cnt[0])
        
        black = np.zeros_like(conImg)
        cv2.drawContours(black, [hull], -1, (255, 255, 255), -1)
        g2 = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        r, t2 = cv2.threshold(g2, 127, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(conImg, conImg, mask = t2)    
        ROI = masked[y:y+h, x:x+w]
        return ROI

    def _check_if_hole(self, roi):
        img=self._get_exact_contour(roi)
        resizedImg = cv2.resize(img, (100,100), interpolation = cv2.INTER_AREA)
        enhImg=self._get_enhanced(resizedImg)
        
        hsvImg= cv2.cvtColor(enhImg, cv2.COLOR_RGB2HSV)  
        
        lower_white = np.array([0,0,168])
        upper_white = np.array([172,111,255])
        whiteMask= cv2.inRange(hsvImg, lower_white, upper_white)
        whiteImg = cv2.bitwise_and(hsvImg,hsvImg, mask= whiteMask)  
        whitePercentage=round((whiteMask>0).mean()*100,2)
        
        lower_black = np.array([0,0,0])
        upper_black = np.array([180,255,40])
        blackMask= cv2.inRange(hsvImg, lower_black, upper_black)
        blackImg = cv2.bitwise_and(hsvImg,hsvImg, mask= blackMask)  
        blackPercentage=round((blackMask>0).mean()*100,2)
        
        if (blackPercentage>whitePercentage 
            and (whitePercentage>=4 and whitePercentage<=15) and blackPercentage<=50):
            #DisplayImage(roi)
            #print('whitePercentage',whitePercentage,'blackPercentage',blackPercentage)
            return True
        else:
            return False

    def _add_text_on_image(self, image, x,y,text):
        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 1.5
        TEXT_THICKNESS = 2
        TEXT = text
        cv2.putText(image, TEXT, (x,y-5), TEXT_FACE, TEXT_SCALE, (0,0,255), TEXT_THICKNESS, cv2.LINE_AA)

    def _extract_colors_from_image(self, img, color):
        ## convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    
        brownColorMask = cv2.inRange(hsv, (10, 100, 20), (30, 255, 200)) #brown color hsv range
        greenColorMask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255)) #green color hsv range
        yellowColorMask = cv2.inRange(hsv, (25, 80, 80), (36, 255, 255)) #yellow color hsv range
        whiteColorMask = cv2.inRange(hsv, (0, 0, 0), (0,0,255)) #white color hsv range
        #whiteColorMask = cv2.inRange(hsv, (0, 0, 0), (35,40,255)) #white color hsv range
        #whiteColorMask = cv2.inRange(hsv, (0, 0, 0), (100,3,100)) #white color hsv range

        if color=="spotscombined":
            mergedMask=brownColorMask
            mergedMask = cv2.bitwise_or(mergedMask, greenColorMask)
            mergedMask = cv2.bitwise_or(mergedMask, yellowColorMask)
        elif color=="leafcombined":
            mergedMask=greenColorMask
            mergedMask = cv2.bitwise_or(mergedMask, yellowColorMask)   
        elif color=="basicplantcombined":
            mergedMask=greenColorMask
            mergedMask = cv2.bitwise_or(mergedMask, yellowColorMask)
        elif color=="brown":
            mergedMask=brownColorMask
        elif color=="green":
            mergedMask=greenColorMask
        elif color=="yellow":
            mergedMask=yellowColorMask
        elif color=="white":
            mergedMask=whiteColorMask

        ## slice the colors by mask
        imask = mergedMask>0
        colorFilteredImage = np.zeros_like(img, np.uint8)
        colorFilteredImage[imask] = img[imask]   
        return colorFilteredImage  

    def _count_and_color_coverage_helper(self, img, color): 
        colorFilteredImage=self._extract_colors_from_image(img,color)
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
            i=0
    
        return colorFilteredImage, colorcoverage,objectcount

    def _count_of_tikka_leaf_spots_and_color_coverage(self, img):         
        colorFilteredImage, combinedcolorcoverage, combinedspotcount = self._count_and_color_coverage_helper(img,"spotscombined")
        colorFilteredImage, browncolorcoverage, spotcount = self._count_and_color_coverage_helper(img,"brown")
        colorFilteredImage, greencolorcoverage, spotcount = self._count_and_color_coverage_helper(img,"green")
        colorFilteredImage, yellowcolorcoverage, spotcount = self._count_and_color_coverage_helper(img,"yellow")
        
        return combinedspotcount, combinedcolorcoverage, browncolorcoverage, greencolorcoverage, yellowcolorcoverage

    def _groundnut_leaf_color_coverage(self, img):         
        colorFilteredImage, combinedcolorcoverage, combinedspotcount = self._count_and_color_coverage_helper(img,"leafcombined")
        colorFilteredImage, greencolorcoverage, spotcount = self._count_and_color_coverage_helper(img,"green")
        colorFilteredImage, yellowcolorcoverage, spotcount = self._count_and_color_coverage_helper(img,"yellow")
        
        return combinedcolorcoverage, greencolorcoverage, yellowcolorcoverage

    def _base64_to_cv2_image(self, base64_str):
        base64_image = base64_str.split(',')[-1]
        img_data = base64.b64decode(base64_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    
    def _cv2_image_to_base64(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

    def _pil_image_to_base64(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")  # You can change the format to PNG or others as needed
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_base64

    def pil_to_cv2(self, pil_image):
        numpy_image = np.array(pil_image)
        
        cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        
        return cv2_image


    def _add_legend_and_counts(self, image, colors, class_counts):
        # image = self._correct_image_orientation(image)
        
        # Calculate font size based on image dimensions
        image_width, image_height = image.size
        base_font_size = max(20, image_width // 30)
        font = ImageFont.truetype("arialbd.ttf", base_font_size)
        
        # Calculate padding dynamically
        padding_vertical = int(image_height * 0.05)  # 5% of image height
        padding_horizontal = int(image_width * 0.05)  # 5% of image width
        
        # Calculate additional height for the legend
        legend_items_count = len(colors)
        additional_height = int(base_font_size * 1.5 * legend_items_count)
        total_height = image_height + additional_height + padding_vertical * 2
        total_width = image_width + padding_horizontal * 2

        # Create new image with padding
        new_image = Image.new('RGB', (total_width, total_height), (0, 0, 0))
        new_image.paste(image, (padding_horizontal, padding_vertical))
        draw = ImageDraw.Draw(new_image)

        # Draw the legend
        y_position = image_height + padding_vertical + 20
        x_position = padding_horizontal + 10

        for class_name, color in colors.items():
            draw.ellipse([x_position, y_position, x_position + base_font_size, y_position + base_font_size], fill=color)
            legend_text = f"{class_name.replace('-', ' ').title()}: {class_counts.get(class_name, 0)}"
            draw.text((x_position + base_font_size + 10, y_position), legend_text, fill="white", font=font)
            y_position += int(base_font_size * 1.5)  # Increase spacing between legend items

        return new_image

    def _detect_from_image(self, cv2_image):
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)

        detection_results = self._sib_img_detection_ultralytics_local(image)

        draw = ImageDraw.Draw(image)

        colors = {
            "gram-pod-borer": "red",
            "tikka-leaf-spot": "blue"
        }

        available_colors = ["blue", "orange", "purple", "pink", "cyan", "lime", "brown", "magenta"]
        used_colors = set(colors.values())

        image_width, image_height = image.size

        class_counts = {}

        for detection in detection_results:
            class_name = detection["name"]
            confidence = detection["confidence"]
            xcenter = detection["xcenter"] * image_width
            ycenter = detection["ycenter"] * image_height
            width = detection["width"] * image_width
            height = detection["height"] * image_height

            xmin = xcenter - width / 2
            ymin = ycenter - height / 2
            xmax = xcenter + width / 2
            ymax = ycenter + height / 2

            # Assign a color if the class is not already in the colors dictionary
            if class_name not in colors:
                if available_colors:
                    color = available_colors.pop(0)
                else:
                    color = "#%06x" % random.randint(0, 0xFFFFFF)  # Generate a random color if we run out of predefined ones
                colors[class_name] = color
            else:
                color = colors[class_name]

            # Draw the bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=5)

            # Update class counts
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        new_image = self._add_legend_and_counts(image, colors, class_counts)
        image = self._pil_image_to_base64(new_image)
        return image
    
    def _sib_img_detection_ultralytics_local(self, pil_image):       
        im1 = pil_image
        results = self.ultralytics_sib_local_model.predict(source=im1, save=False)
        detections = results[0].boxes
        names = results[0].names
        width, height = im1.size

        detection_results = []

        for detection in detections:
            class_id = int(detection.cls.item())
            class_name = names[class_id]
            confidence = detection.conf.item()
            bbox = detection.xyxy[0]
            xmin, ymin, xmax, ymax = bbox

            xcenter = (xmin + xmax) / 2
            ycenter = (ymin + ymax) / 2
            width_bbox = xmax - xmin
            height_bbox = ymax - ymin

            detection_results.append({
                "name": class_name,
                "confidence": float(confidence),
                "xcenter": xcenter / width,
                "ycenter": ycenter / height,
                "width": width_bbox / width,
                "height": height_bbox / height
            })
        
        return detection_results
