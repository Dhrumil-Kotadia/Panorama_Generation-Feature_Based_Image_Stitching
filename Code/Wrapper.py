#!/usr/bin/evn python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from skimage.feature import peak_local_max
from glob import glob
from argparse import ArgumentParser

def Corner_Detection(Im, Im_Gray):
    """
    Corner Detection using Harris Corner Detection and Good Features to Track

    Input:
    Im -> Image
    Im_Gray -> Grayscale Image

    Output:
    Im_Corner_Harris -> Image with Corners Marked
    Corners -> Corners Detected
    temp_Img -> Image with Corners Marked
    """
    # # Smaller Block size can detect small corners but is sensitive to noise.
    # # Smaller KSize means it is more sensitive to noise but does not miss the finer details
    # # Smaller K makes the algorithm less strict to determine what is a corner
    
    Im_Corner_Harris = cv2.cornerHarris(Im_Gray,2,1,0.01)

    Corners = np.where(Im_Corner_Harris > 0.001 * Im_Corner_Harris.max())
    No_Corners = Im_Corner_Harris<0.001 * Im_Corner_Harris.max()
    Im_Corner_Harris[No_Corners] = 0
    
    Corners_x, Corners_y = Corners
    Corners_x = Corners_x.tolist()
    Corners_y = Corners_y.tolist()
    
    Corners = []
    for i in range(len(Corners_x)):
        Corners.append([Corners_x[i],Corners_y[i]])
    
    ################################################################
    # Good Features To Track Implementation
    # Corners_GFTT = cv2.goodFeaturesToTrack(image=Im_Gray,maxCorners=10000,qualityLevel=0.00001,minDistance=10)
    # print("Corners GFTT",Corners_GFTT)
    # print("Corners GFTT Shape:",Corners_GFTT.shape)
    # Corners = []
    # Corners_x=[]
    # Corners_y=[]
    # for i in Corners_GFTT:
    #     Corners_x.append(int(i[0][0]))
    #     Corners_y.append(int(i[0][1]))
    #     Corners.append([int(i[0][0]),int(i[0][1])])
    # # print('Corners_X:',Corners_x)
    # # print('Corners_Y:',Corners_y)
    # # print('Corners:',Corners)
    # Im_Corners_GFTT1 = copy.deepcopy(Im)
    ################################################################
    
    temp_Img = copy.deepcopy(Im)
    for i in range(len(Corners)):
        cv2.circle(temp_Img,(Corners[i][1],Corners[i][0]),1,(0,0,255),1)

    return Im_Corner_Harris, Corners, temp_Img

def ANMS(Im,Im_Gray,Corners,n_best):
    """
    Perform Adaptive Non-Maximal Suppression for Corner Detection

    Input:
    Im -> Image
    Im_Gray -> Grayscale Image
    Corners -> Corners Detected
    n_best -> Number of Best Corners

    Output:
    Im_Corners_GFTT2 -> Image with Corners Marked
    Valid_Corners -> Valid Corners
    """
    Corners_GFTT = Corners
    Im_Corners_GFTT2 = copy.deepcopy(Im)
    Length = len(Corners_GFTT)
    
    R = np.full((Length,3),100000,int)
    ED = 0
    
    for i in range(len(Corners_GFTT)):
    # for i in range(200):
        [xi,yi] = Corners_GFTT[i]
        xi = int(xi)
        yi = int(yi)
        for j in range(len(Corners_GFTT)):
            [xj,yj] = Corners_GFTT[j]
            xj = int(xj)
            yj = int(yj)
            if Im_Gray[xj,yj] > Im_Gray[xi,yi]:
                ED = (xj-xi)**2 + (yj-yi)**2
            if ED < R[i][0]:
                R[i][0] = ED
                R[i][1] = xi
                R[i][2] = yi
    
    R = R[R[:, 0].argsort()]
    R = np.flip(R,0)
    Valid_Corners = []
    for i in range(n_best):
        X = R[i][1]
        Y = R[i][2]
        if int(X) != 100000:
            Valid_Corners.append([Y,X])
            cv2.circle(Im_Corners_GFTT2,(Y,X),1,(0,0,255),1)
    
    return Im_Corners_GFTT2, Valid_Corners

def Feature_Descriptor(Im_Gray,Valid_Corners):
    """
    Generate feature descriptors

    Input:
    Im_Gray-> Grayscale Image
    Valid_Corners -> Valid Corners

    Output:
    Features -> Feature Descriptors
    """
    Im_Gray = np.pad(Im_Gray,[(20,20),(20,20)])
    # plt.figure()
    Features = []
    for i in Valid_Corners:
        patch = Im_Gray[(i[1]):(i[1]+41),(i[0]):(i[0]+41)]
        
        Blurred_Patch = cv2.GaussianBlur(patch,(5,5),0)
        Feature = cv2.resize(Blurred_Patch,(8,8))
        # plt.subplot((len(Valid_Corners)//10)+1,10,i)
        Feature = np.reshape(Feature,(64,1))
        
        Mean = np.mean(Feature)
        SD = np.std(Feature)
        Feature = (Feature-Mean)/SD
        Features.append(Feature)
    # plt.show()
    return Features

def Feature_Matching(Features1, Features2, Valid_Corners1, Valid_Corners2):
    """
    Feature Matching using Euclidean Distance

    Input:
    Features1 -> Features of Image 1
    Features2 -> Features of Image 2
    Valid_Corners1 -> Valid Corners in Image 1
    Valid_Corners2 -> Valid Corners in Image 2

    Output:
    Pairs -> Matching Pairs
    Final_Distances -> Final Distances
    """
    Final_Distances = []
    Pairs = []
    for i in range(len(Features1)):
        Distances = []
        Distance_Points = {}
        for j in range(len(Features2)):
            Distance = np.linalg.norm(Features1[i]-Features2[j])**2
            Distances.append(Distance)
            Distance_Points[Distance] = [Valid_Corners1[i],Valid_Corners2[j]]
            
        
        Distances.sort()
        
        Ratio = Distances[0]/Distances[1]
        if Ratio < 0.7:
            Pairs.append(Distance_Points[Distances[0]])
            Final_Distances.append(Distances[0])
            
    return Pairs, Final_Distances

def draw_Matches(Im1,Im2,Valid_Keypoints1,Valid_Keypoints2):
    """
    Draw Matches between two images

    Input:
    Im1 -> Image 1
    Im2 -> Image 2
    Valid_Keypoints1 -> Valid Keypoints in Image 1
    Valid_Keypoints2 -> Valid Keypoints in Image 2

    Output:
    Match_Img -> Image with Matches
    """
    # print("Im1 Shape:",Im1.shape)
    # print("Im2 Shape:",Im2.shape)
    if Im1.shape[0] != Im2.shape[0]:
        Required_height = max(int(Im1.shape[0]),int(Im2.shape[0]))
        if Im1.shape[0] > Im2.shape[0]:
            Im2 = cv2.copyMakeBorder(Im2,0,Required_height-Im2.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
            # print("New Im2 Shape:",Im2.shape)
        elif Im1.shape[0] < Im2.shape[0]:
            Im1 = cv2.copyMakeBorder(Im1,0,Required_height-Im1.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
            # print("New Im1 Shape:",Im1.shape)
    Match_Img = cv2.hconcat([Im1,Im2])
    # H,W,C = Im2.shape
    for i in Valid_Keypoints1:
        cv2.circle(Match_Img,(i[0],i[1]),1,(0,0,255),1)
    for i in Valid_Keypoints2:
        cv2.circle(Match_Img,(Im1.shape[1]+i[0],i[1]),1,(0,0,255),1)
    for i in range(len(Valid_Keypoints1)):
        cv2.line(Match_Img,(Valid_Keypoints1[i][0],Valid_Keypoints1[i][1]),(Im1.shape[1]+Valid_Keypoints2[i][0],Valid_Keypoints2[i][1]),(0,255,0),1)
    
    return Match_Img

def Calculate_Homography(Points1, Points2, num=4):
    """
    Calculate Homography

    Input:
    Points1 -> Points in Image 1
    Points2 -> Points in Image 2
    num -> Number of Points

    Output:
    H -> Homography Matrix
    """
    A = np.zeros((2*num,9))
    X1 = np.zeros(num,dtype=int)
    X2 = np.zeros(num,dtype=int)
    Y1 = np.zeros(num,dtype=int)
    Y2 = np.zeros(num,dtype=int)

    for i in range(num):
        X1[i] = Points1[i][0]
        X2[i] = Points2[i][0]
        Y1[i] = Points1[i][1]
        Y2[i] = Points2[i][1]
    for i in range(num):
        
        A[2*i,:] = [0,0,0,-X1[i],-Y1[i],-1,Y2[i]*X1[i],Y2[i]*Y1[i],Y2[i]]
        A[2*i+1,:] = [X1[i],Y1[i],1,0,0,0,-X2[i]*X1[i],-X2[i]*Y1[i],-X2[i]]
    
    _,_,V = np.linalg.svd(A)
    H = V[-1,:].reshape(3,3)
    H = H/H[2,2]

    return H

def Ransac(Valid_Keypoints1,Valid_Keypoints2,Iter):
    """
    Outlier Filtering using RANSAC

    Input:
    Valid_Keypoints1 -> Valid Keypoints in Image 1
    Valid_Keypoints2 -> Valid Keypoints in Image 2
    Iter -> Number of Iterations

    Output:
    H_Final -> Final Homography Matrix
    Final_Indices -> Final Inliers
    """
    Max_Inliers = 0
    Final_Indices = []

    for iterations in range(Iter):

        Rand_Index = [np.random.randint(0,len(Valid_Keypoints1))for i in range(4)]
        Pts1 = [Valid_Keypoints1[i] for i in Rand_Index]
        Pts2 = [Valid_Keypoints2[i] for i in Rand_Index]
        H = Calculate_Homography(Pts1,Pts2)
        
        Index = []
        inliers = 0
        for i in range(len(Valid_Keypoints1)):
            [x,y] = Valid_Keypoints1[i]
            [u,v] = Valid_Keypoints2[i]
            Temp = np.dot(H,np.array([x,y,1]))
            if Temp[2] != 0:
                Temp_x = Temp[0]/Temp[2]
                Temp_y = Temp[1]/Temp[2]
            elif Temp[2] == 0:
                Temp_x = Temp[0]/0.00001
                Temp_y = Temp[0]/0.00001
            Temp = [Temp_x,Temp_y]

            Distance = np.linalg.norm(np.array(Temp)-np.array([u,v]))**2

            if Distance < 30:
                inliers += 1
                Index.append(i)
            if inliers > Max_Inliers:
                Max_Inliers = inliers
                Final_Indices = Index
                H_Final = H
            if inliers > 0.9*len(Valid_Keypoints1):
                Final_Indices = Index
                H_Final = H
                break


    return H_Final, Final_Indices

def RemoveBG(image):
    """
    Remove the background of the image using a simple thresholding technique

    Input:
    image -> Image

    Output:
    Mask -> Mask of the Foreground
    """
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY)
    Mask = cv2.medianBlur(thresh,5)
    
    return Mask

def warpTwoImages(img1, img2, H):
    """
    Warp two images with homography

    Input:
    img1 -> Image 1
    img2 -> Image 2
    H -> Homography Matrix

    Output:
    result -> Blended Image
    """
    # warp two images with homography H. This code is inspired from various sources on the internet.
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    points1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    points2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    Temp = cv2.perspectiveTransform(points2, H)
    pts = np.concatenate((points1, Temp), axis=0)
    
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    Temp = [-xmin,-ymin]
    Ht = np.array([[1,0,Temp[0]],[0,1,Temp[1]],[0,0,1]]) # translate
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))

    # Compute Mask of Foreground and paste it onto the Background Image
    Mask  = RemoveBG(img1)
    overlap = result[Temp[1]:h1+Temp[1],Temp[0]:w1+Temp[0]]
    BG = cv2.bitwise_and(overlap.copy(),overlap.copy(),mask = cv2.bitwise_not(Mask))
    FG = cv2.bitwise_and(img1.copy(),img1.copy(),mask = Mask)
    result[Temp[1]:h1+Temp[1],Temp[0]:w1+Temp[0]] = cv2.add(BG, FG)
    
    print("Warping Done!")
    return result

def CreatePanorama(Im1, Im2, i):
    """
    Read a set of images for Panorama stitching
    """
    
    Im1_Gray = cv2.cvtColor(Im1,cv2.COLOR_BGR2GRAY)
    Im2_Gray = cv2.cvtColor(Im2,cv2.COLOR_BGR2GRAY)

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    Im1_Corners,Corners1,corners_Img = Corner_Detection(Im1,Im1_Gray)
    cv2.imwrite(str(i)+'_Image1_corners'+'.png',corners_Img)
    Im2_Corners,Corners2,corners_Img = Corner_Detection(Im2,Im2_Gray)
    cv2.imwrite(str(i)+'_Image2_corners'+'.png',corners_Img)

    
    NS1 = peak_local_max(Im1_Corners,min_distance=10).shape[0]
    NS2 = peak_local_max(Im2_Corners,min_distance=10).shape[0]
    
    N_Best = min(NS1,NS2)
    
    print("Number of best corners: "+str(N_Best))
    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    ## Valid Corners - Contains the N-Best Corners
    ## ANMS_Out - Image with Corners Marked
    ANMS_Out1, Valid_Corners1 = ANMS(Im1,Im1_Gray,Corners1,N_Best)
    cv2.imwrite(str(i)+'_Image1_anms'+'.png',ANMS_Out1)
    ANMS_Out2, Valid_Corners2 = ANMS(Im2,Im2_Gray,Corners2,N_Best)
    cv2.imwrite(str(i)+'_Image2_anms'+'.png',ANMS_Out2)

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    Features1 = Feature_Descriptor(Im1_Gray,Valid_Corners1)
    Features2 = Feature_Descriptor(Im2_Gray,Valid_Corners2)
    
    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    
    Match_pairs,Distances = Feature_Matching(Features1, Features2, Valid_Corners1, Valid_Corners2)
    Valid_Keypoints1 = []
    Valid_Keypoints2 = []
    DMatch_Pairs = []
    for j in Match_pairs:
        Valid_Keypoints1.append(j[0])
        Valid_Keypoints2.append(j[1])
    Distances = np.array(Distances)
    
    # If less matching features, ignore the image
    if len(Match_pairs) < 5:
        print('Not enough matching features.')
        print('----')
        return 0
    
    #Using custom function for drawing matches. (Professor mentioned that it is okay to use a custom function for drawging matches)
    Match_Img = draw_Matches(Im1,Im2,Valid_Keypoints1,Valid_Keypoints2)
    cv2.imwrite(str(i)+'_matching.png',Match_Img)

    """
    Refine: RANSAC, Estimate Homography
    """
    H_Keypoints1 = []
    H_Keypoints2 = []
    Iter = 5000 # Number of iterations for which RANSAC will compute the homography using 4 random points
    H, Index = Ransac(Valid_Keypoints1,Valid_Keypoints2,Iter)
    
    for j in Index:
        H_Keypoints1.append(Valid_Keypoints1[j])
        H_Keypoints2.append(Valid_Keypoints2[j])
    

    Ransac_Output = draw_Matches(Im1,Im2,H_Keypoints1,H_Keypoints2)
    cv2.imwrite(str(i)+'_Ransac_Output.png',Ransac_Output)
    
    Dummy1 = []
    Dummy2 = []
    for Key in range(len(H_Keypoints1)):
        if H_Keypoints1[Key] not in Dummy1:
            Dummy1.append(H_Keypoints1[Key])
        if H_Keypoints2[Key] not in Dummy2:
            Dummy2.append(H_Keypoints2[Key])
    # If one corner is matched to multiple corners, skip image
    Diff = abs(len(Dummy1)-len(Dummy2))
    Ratio = Diff/max(len(Dummy1),len(Dummy2))
    if Ratio > 0.3:
        print('Improper Matching')
        print(len(Dummy1))
        print(len(Dummy2))
        print(1-Ratio)
        return 0
    

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """
    # Create Homography Matrix Again using all valid keypoints to mitigate minor errors in the homography
    H = Calculate_Homography(H_Keypoints1,H_Keypoints2,len(Index))
    
    blended_result = warpTwoImages(Im2, Im1, H)
    cv2.imwrite('mypano'+'.png',blended_result)
    
def main():
    
    # Parse Args to obtain Image Set and the Method to be used for Panorama Generation
    Parser = ArgumentParser()
    Parser.add_argument('--Set', default=2, help='The Set of Images to be used. Choose 1,2 or 3')
    Parser.add_argument('--Method', default=1, help='The Set of Images to be used. Choose 1 or 2')

    Args = Parser.parse_args()
    Image_Set = Args.Set
    Method = Args.Method
    Image_path = '../Data/Test/TestSet'+str(Image_Set)+'/*.jpg'

    # Get all images and sort paths
    All_Images = sorted(glob(Image_path))
    print('All Images', All_Images)
    print(Method)
    if int(Method) == 1:
        for i in range(len(All_Images)):
            print('i:',i)
            if i == len(All_Images)-1:
                print('Breaking!')
                break
            if i == 0:
                path1 = All_Images[i]
                Im1 = cv2.imread(path1)
                path2 = All_Images[i+1]
                Im2 = cv2.imread(path2)
                CreatePanorama(Im1, Im2, i)
            else:
                path2 = All_Images[i+1]
                Im1 = cv2.imread(path2)
                Im2 = cv2.imread('mypano'+'.png')
                CreatePanorama(Im1, Im2, i)

    elif int(Method) == 2:
        Counter = 1
        Images_Used_Counter = 1
        im = 0
        Skip_Flag = False
        if len(All_Images)%2 == 0:
            centre_index_1 = len(All_Images)//2 - 1
            centre_index_2 = (len(All_Images)//2)
            CreatePanorama(cv2.imread(All_Images[centre_index_1]), cv2.imread(All_Images[centre_index_2]), centre_index_1)
            for count in range(len(All_Images)):
                if Skip_Flag == True:
                    Skip_Flag = False
                    continue
                if count != centre_index_1:
                    path1 = All_Images[count]
                    Image = cv2.imread(path1)
                    cv2.imwrite('../temp_data/'+str(count)+'.jpg',Image)
                    
                elif count == centre_index_1:
                    Image = cv2.imread('mypano.png')
                    cv2.imwrite('../temp_data/'+str(count)+'.jpg',Image)
                    Skip_Flag = True
            
            Image_path = '../temp_data/'+'*.jpg'
            All_Images = sorted(glob(Image_path))
            Centre_Index = (len(All_Images)//2)


        elif len(All_Images)%2 != 0:
            Centre_Index = (len(All_Images)//2)
        for i in range(len(All_Images)//2):
            

            if i == 0:
                path1 = All_Images[Centre_Index]
                Im1 = cv2.imread(path1)
                path2 = All_Images[Centre_Index-Counter]
                Im2 = cv2.imread(path2)
                CreatePanorama(Im1, Im2, Images_Used_Counter)
                Images_Used_Counter += 1
                Im2 = cv2.imread('mypano'+'.png')
                path2 = All_Images[Centre_Index+Counter]
                Im1 = cv2.imread(path2)
                CreatePanorama(Im1, Im2, Images_Used_Counter)
                Counter += 1
                Images_Used_Counter += 1
                if Images_Used_Counter == len(All_Images):
                    break
            else:
                Im2 = cv2.imread('mypano'+'.png')
                path2 = All_Images[Centre_Index-Counter]
                Im1 = cv2.imread(path2)
                CreatePanorama(Im1, Im2, Images_Used_Counter)
                Images_Used_Counter += 1
                Im2 = cv2.imread('mypano'+'.png')
                path2 = All_Images[Centre_Index+Counter]
                Im1 = cv2.imread(path2)
                CreatePanorama(Im1, Im2, Images_Used_Counter)
                Counter += 1
                Images_Used_Counter += 1
                if Images_Used_Counter == len(All_Images):
                    break
        
if __name__ == "__main__":
    main()
