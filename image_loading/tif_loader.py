img_filename = DATA_DIR + "/train-tif-v2/train_5761.tif"
tif_img = io.imread(img_filename)[:,:,:3]
plt.imshow(tif_img)
