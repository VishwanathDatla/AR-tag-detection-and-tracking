
import numpy as np
from numpy import linalg as la
import math
import random
import cv2


#-------------------------------------------------------------------------------

# @brief Function for converting gray scale to binary
#
#  @param Matrix
#
#  @return Matrix
#
def binary(im):
    for row in range(0,len(im)):
        for col in range(0,len(im[0])):
            if (im[row,col]>150):
                im[row,col]=1
            else:
                im[row,col]=0
    return im

# @brief To give 4 points in Edgedetection
#
#  @param Image and old_ctr points
#
#  @return corners
#
def Edge(image,old_ctr):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,3)
    (t, thres) = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    contours, hierarchy=cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr=[]
    for i, cont in zip(hierarchy[0], contours):
        cnt_len = cv2.arcLength(cont,True)
        cont = cv2.approxPolyDP(cont, 0.02*cnt_len,True)
        if cv2.contourArea(cont) > 1000 and cv2.isContourConvex(cont) and len(cont) == 4  :
            cont=cont.reshape(-1,2)
            if i[0] == -1 and i[1] == -1 and i[3] != -1:
                ctr.append(cont)
        #print(np.shape(ctr))
        old_ctr=ctr
    return ctr

# @brief Calculate homography
#
#  @param source and destination
#
#  @return homography
#
def homography_calc(src,desi):
    #print(des)
    #print(np.shape(des))
    c1 = desi[0,0]
    c2 = desi[0,1]
    c3 = desi[0,2]
    c4 = desi[0,3]

    w1 = src[0]
    w2 = src[1]
    w3 = src[2]
    w4 = src[3]

    A=np.array([[w1[0],w1[1],1,0,0,0,-c1[0]*w1[0],-c1[0]*w1[1],-c1[0]],
                [0,0,0,w1[0], w1[1],1,-c1[1]*w1[0],-c1[1]*w1[1],-c1[1]],
                [w2[0],w2[1],1,0,0,0,-c2[0]*w2[0],-c2[0]*w2[1],-c2[0]],
                [0,0,0,w2[0], w2[1],1,-c2[1]*w2[0],-c2[1]*w2[1],-c2[1]],
                [w3[0],w3[1],1,0,0,0,-c3[0]*w3[0],-c3[0]*w3[1],-c3[0]],
                [0,0,0,w3[0], w3[1],1,-c3[1]*w3[0],-c3[1]*w3[1],-c3[1]],
                [w4[0],w4[1],1,0,0,0,-c4[0]*w4[0],-c4[0]*w4[1],-c4[0]],
                [0,0,0,w4[0], w4[1],1,-c4[1]*w4[0],-c4[1]*w4[1],-c4[1]]])

    #Performing SVD
    u, s, vt = la.svd(A)

            # normalizing by last element of v
            #v =np.transpose(v_col)
    v = vt[8:,]/vt[8][8]

    v_req = np.reshape(v,(3,3))

    return v_req
# @brief To fix image on top of AR tag
#
#  @param image , corners , source
#
#  @return superimposed image
#
def warp_perspective(image, h_matrix, dimension):
    
    warped = np.zeros((dimension[0], dimension[1], 3))
    for index1 in range(0, image.shape[0]):
        for index2 in range(0, image.shape[1]):
            new_vec = np.dot(h_matrix, [index1, index2, 1])
            new_row, new_col, _ = (new_vec / new_vec[2] + 0.4).astype(int)
            if 5 < new_row < (dimension[0] - 5):
                if 5 < new_col < (dimension[1] - 5):
                    warped[new_row, new_col] = image[index1, index2]
                    warped[new_row - 1, new_col - 1] = image[index1, index2]
                    warped[new_row - 2, new_col - 2] = image[index1, index2]
                    warped[new_row - 3, new_col - 3] = image[index1, index2]
                    warped[new_row + 1, new_col + 1] = image[index1, index2]
                    warped[new_row + 2, new_col + 2] = image[index1, index2]
                    warped[new_row + 3, new_col + 3] = image[index1, index2]

    return np.array(warped, dtype=np.uint8)
    #Transpose the image 



def superimpose(cont,image,src):
    pts_dst = np.array(cont,dtype=float)
    pts_src = np.array([[0,0],[511, 0],[511, 511],[0,511]],dtype=float)
    H = homography_calc(pts_src, pts_dst)


    tmp = cv2.warpPerspective(src, H,(image.shape[1],image.shape[0]));
    cv2.fillConvexPoly(image, pts_dst.astype(int), 0, 16);

    image = image + tmp;

    return image,H
# @brief To find get grid of tag image
#
#  @param image , corners
#
#  @return tag image and gray scale image
#


def computeHomography(c1, c2):
    if (len(c1) < 4) or (len(c2) < 4):
        print("Need atleast four points to compute SVD.")
        return 0
    x = c1[:, 0]
    y = c1[:, 1]
    xp = c2[:, 0]
    yp = c2[:,1]
    nrows = 8
    ncols = 9  
    A = []
    for i in range(int(nrows/2)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)
    A = np.array(A)
    U, E, VT = np.linalg.svd(A)
    V = VT.transpose()
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]
    return H



def perspective(ctr,image):
    d = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]], dtype = "float32")

    M1 = computeHomography(ctr[0], d)
    w1 = cv2.warpPerspective(image.copy(), M1, (100,100))
    w2=cv2.medianBlur(w1,3)
    #warp2= warp1-warp1_5

    tag_image=cv2.resize(w2, dsize=None, fx=0.08, fy=0.08)
    return tag_image,w2
# @brief To find tag ID and tag image
#
#  @param image , corners
#
#  @return tag and tag_id
#
def Tag_value(cont,tag_image):
    gray = cv2.cvtColor(tag_image,cv2.COLOR_BGR2GRAY)
    pixel=binary(gray)
   # print(pixel_value,'Tag Value')
    status=0
    A_cont=cont[0][0]
    #print(A_ctr,'ctr A')
    B_cont=cont[0][1]
    #print(B_ctr,'ctr B')
    C_cont=cont[0][2]
    #print(C_ctr,'ctr B')
    D_cont=cont[0][3]
    #print(D_ctr,'ctr C')
    if (pixel[2,2] == 1):
        L1=A_cont
        L2=B_cont
        L3=C_cont
        L4=D_cont
        status=0
        o = pixel[4,4]
        t = pixel[4,3]
        th = pixel[3,3]
        f = pixel[3,4]

    elif pixel[5,2]==1:
        L1=D_cont
        L2=A_cont
        L3=B_cont
        L4=C_cont
        status=1
        o = pixel[3,4]
        t = pixel[4,4]
        th = pixel[4,3]
        f = pixel[3,3]

    elif pixel[5,5] == 1:
        L1=C_cont
        L2=D_cont
        L3=A_cont
        L4=B_cont
        status=2
        o = pixel[3,3]
        t = pixel[3,4]
        th = pixel[4,4]
        f = pixel[4,3]

    elif pixel[2,5] == 1:
        L1=B_cont
        L2=C_cont
        L3=D_cont
        L4=A_cont
        status=3
        o = pixel[4,3]
        t = pixel[3,3]
        th = pixel[3,4]
        f = pixel[4,4]

    else:
        L1=A_cont
        L2=B_cont
        L3=C_cont
        L4=D_cont
        o = pixel[4,4]
        t = pixel[4,3]
        th = pixel[3,3]
        f = pixel[3,4]


    new_cont=np.array([[L1,L2,L3,L4]])

    #print(new_ctr,'new_ctr')

    tag_id = f*8 + th*4 + t*2 + o*1
    print('Tag value is',tag_id)
    return new_cont,tag_id
# @brief To draw 3d cube
#
#  @param image , perspective points in camera frame
#
#  @return image of cube
#
def draw(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpoints[:4]],-1,(255,0,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpoints[i]), tuple(imgpoints[j]),(0,150,150),3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpoints[4:]],-1,(0,0,255),3)
    return img

# @brief Calculates projection matrix
#
#  @param Homography
#
#  @return Projection matrix for 3D transformation
#
def Projection_mat(h):
   # homography = homography*(-1)
   # Calling the projective matrix function
   k = np.array([[1346.1005953,0,932.163397529403],[0, 1355.93313621175,654.898679624155],[0, 0,1]])
   K =np.array([[1406.08415449821,0,0],[ 2.20679787308599, 1417.99930662800,0],[ 1014.13643417416, 566.347754321696,1]])
   K=K.T
   rot_tran = np.dot(np.linalg.inv(K), h)
   col1 = rot_tran[:, 0]
   col2 = rot_tran[:, 1]
   col3 = rot_tran[:, 2]
   l = math.sqrt(np.linalg.norm(col1, 2) * np.linalg.norm(col2, 2))
   rot_1 = col1 / l
   rot_2 = col2 / l
   translation = col3 / l
   c = rot_1 + rot_2
   p = np.cross(rot_1, rot_2)
   d = np.cross(c, p)
   rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
   rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
   rot_3 = np.cross(rot_1, rot_2)

   projection = np.stack((rot_1, rot_2, rot_3, translation)).T
   return np.dot(K, projection)

    #def Three_d_cube(K, homography):
    #    return 0
# @brief it draws cube on image
#
#  @param Projection matrix and iamge
#
#  @return None
#
def Cube(proj_mat,image):
    axis = np.float32([[0,0,0,1],[0,512,0,1],[512,512,0,1],[512,0,0,1],[0,0,-512,1],[0,512,-512,1],[512,512,-512,1],[512,0,-512,1]])
    Proj= np.matmul(axis,proj_mat.T)
    # Normalize the matrix
    Norm1 = np.divide(Proj[0],Proj[0][2])
    Norm2 = np.divide(Proj[1],Proj[1][2])
    Norm3 = np.divide(Proj[2],Proj[2][2])
    Norm4 = np.divide(Proj[3],Proj[3][2])
    Norm5 = np.divide(Proj[4],Proj[4][2])
    Norm6 = np.divide(Proj[5],Proj[5][2])
    Norm7 = np.divide(Proj[6],Proj[6][2])
    Norm8 = np.divide(Proj[7],Proj[7][2])

    points = np.vstack((Norm1,Norm2,Norm3,Norm4,Norm5,Norm6,Norm7,Norm8))
    final_2d=np.delete(points,2, axis=1)
    draw(image,final_2d)
    return image
# @brief Main loop to run the code and video breaking
#
#  @param path of video, source of lena
#
#  @return image_array and size of image
#
def Imageprocessor(path,src):

    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    img_array=[]

    while (success):
        if (count==0):
            success, image = vidObj.read()
        height,width,layers=image.shape

        size = (width,height)
        if (count==0):
            old_corners=0
        corners=Edge(image,old_corners)
        if(len(corners)==0):
            corners=old_corners

        tag_image,Tag=perspective(corners,image)
        new_corners,tag_id=Tag_value(corners,tag_image)


        image,h=superimpose(new_corners,image,src)
        proj_mat=Projection_mat(h)
       # image=Cube(proj_mat,image)
        old_corners=corners
        count += 1
        print('Number of frames is',count)
        cv2.imwrite('%d.jpg' %count,image)
        img_array.append(image)
        success, image = vidObj.read()

    return img_array,size
#--------------------------------------------------------------
# @brief Loop to run video from image array
#
#  @param final image array and size
#
#  @return None
#

#---------------------------------------------------------------
# main
if __name__ == '__main__':

    # Calling the function
    src=cv2.imread('testudo.png')
    #print(np.size(src))
    Image,size=Imageprocessor('1tagvideo.mp4',src)
    video=cv2.VideoWriter('video_cube.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
    for i in range(len(Image)):
        video.write(Image[i])
    video.release()
    
    