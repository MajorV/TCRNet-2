def crop(input_img,d1,d2):
    '''
    This function returns a cropped image.

    input_img  = input image
    d1 = rows of cropped image
    d2 = column of cropped image
    '''
    m,n = input_img.shape

    off1 = round((m-d1)/2)
    off2 = round((n-d2)/2) -1

    if off1 < 0:
        off1 = 0

    if off2 < 0:
        off2 = 0

    return input_img[off1:off1+d1,off2:off2+d2]
