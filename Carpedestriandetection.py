#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import tensorflow as tf
from scipy.misc import imread
import matplotlib.pyplot as plt

import operator
import itertools
import pickle
from ssd import SSD300
from ssd_utils import BBoxUtility

from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score,                             classification_report, precision_recall_curve, average_precision_score

from keras.utils import np_utils
from keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
from keras.applications.imagenet_utils import preprocess_input

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['image.interpolation'] = 'nearest'

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.45
# set_session(tf.Session(config=config))


# In[6]:


NUM_CLASSES = 3 # 1 added to include background
CLASSES = ['car', 'pedestrian']
input_shape = (300, 300, 3)


# In[7]:


gt = pickle.load(open('groundtruth-encoded.pkl', 'rb'))
base_path = 'checkpoints-nobackground/'
# predictions = pickle.load(open(base_path + 'predictions.pkl', 'rb'))
priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)


# In[8]:


# format [xmin, ymin, xmax, ymax, encoded_y]
gt['object-detection-crowdai/1479505369437332886.jpg']


# In[5]:


predictions


# In[4]:


pred_prob = predictions['pred_prob']
train_keys = predictions['train_keys']
val_keys = predictions['val_keys']


# In[5]:


def detection_out(predictions, background_label_id=0, keep_top_k=200, confidence_threshold=0.01):
    mbox_loc = predictions[:, :, :4]
    variances = predictions[:, :, -4:]
    mbox_priorbox = predictions[:, :, -8:-4]
    mbox_conf = predictions[:, :, 4:-8]
    results = []
    scores = []
#     print('len(mbox_loc)', len(mbox_loc))
#     print('len(variances)', len(variances))
#     print('len(mbox_priorbox)', len(mbox_priorbox))
#     print('len(mbox_conf)', len(mbox_conf))
    
    for i in range(len(mbox_loc)):
        results.append([])
        scores.append([])
        decode_bbox = bbox_util.decode_boxes(mbox_loc[i],
                                        mbox_priorbox[i], variances[i])
        
        for c in range(bbox_util.num_classes):
            if c == background_label_id:
                continue
            c_confs = mbox_conf[i, :, c]
            all_confs = mbox_conf[i, :, :]
            c_confs_m = c_confs > confidence_threshold
            
            if len(c_confs[c_confs_m]) > 0:
                boxes_to_process = decode_bbox[c_confs_m]
                confs_to_process = c_confs[c_confs_m]
                all_confs_to_process = all_confs[c_confs_m]
                
                feed_dict = {bbox_util.boxes: boxes_to_process,
                             bbox_util.scores: confs_to_process}
                idx = bbox_util.sess.run(bbox_util.nms, feed_dict=feed_dict)
                good_boxes = boxes_to_process[idx]
                confs = confs_to_process[idx][:, None]
    
                equal = np.all(mbox_conf[i, :, :] == mbox_conf[i, ::])
                if not equal:
                    raise ValueError('FUCKKKKKKKKKKKKK')
                                 
                labels = c * np.ones((len(idx), 1))
                c_pred = np.concatenate((labels, confs, good_boxes),
                                        axis=1)

                scores[-1].extend(all_confs_to_process[idx])
                results[-1].extend(c_pred)
            
        if len(results[-1]) > 0:
            results[-1] = np.array(results[-1])
            scores[-1] = np.array(scores[-1])
            argsort = np.argsort(results[-1][:, 1])[::-1]
            results[-1] = results[-1][argsort]
            results[-1] = results[-1][:keep_top_k]
            scores[-1] = scores[-1][argsort]
            scores[-1] = scores[-1][:keep_top_k]

#         print('final scores[-1]', scores[-1])
#         print('final results[-1]', results[-1])
    return results, scores

results, scores = detection_out(pred_prob, keep_top_k=50, confidence_threshold=0.7)


# In[8]:


# format: [label, confidence, bounding box points]
results


# In[9]:


len(results[0])


# In[10]:


# probability of each class, 0 indicating probability of background
scores


# In[11]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1), decimals=4)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[5]:


def buildY(results):
    y_true = []
    y_pred = []

    for i, k in enumerate(val_keys):
        y_onehot = gt[k][:, 4:]
        y = np.argmax(y_onehot, axis=1)

        result = results[i]
        if len(result) > 0:
            pred = result[:, 0].astype(int)
            pred -= 1 # label starts at 0, ignoring background label

            size = min(len(pred), len(y))
            y_true.extend(y[:size])
            y_pred.extend(pred[:size])
            
    return y_true, y_pred

def buildYWithProb(results, scores):
    less_pred_count = 0
    more_pred_count = 0
    y_true = []
    y_pred = []
    y_scores = []
    for i, k in enumerate(val_keys):
        y_onehot = gt[k][:, 4:]
        c = np.zeros((len(y_onehot), 1))
        y_onehot = np.concatenate((c, y_onehot), axis=1) # add 0 for background
        y = np.argmax(y_onehot, axis=1).tolist()

        score = scores[i]
        pred = results[i]

        pred_length = len(pred)
        y_length = len(y)

        if pred_length > 0:
            pred = pred[:, 0].astype(int)

        if pred_length > y_length:
            more_pred_count += 1
            z = np.zeros(pred_length, dtype=int)
            z[:y_length] = y
            y = z
        elif pred_length < y_length:
            less_pred_count += 1
            z = np.zeros(y_length, dtype=int)
            z_scores = np.array([[0] * 4] * y_length, dtype=float)
            if pred_length != 0:
                z_scores[:pred_length] = score
            z[:pred_length] = pred
            score = z_scores
            pred = z

        y_true.extend(y)
        y_pred.extend(pred)
        y_scores.extend(score)
    
    return y_true, y_pred, np.array(y_scores)
    


# ## Confustion Matrix

# In[13]:


results, scores = detection_out(pred_prob, keep_top_k=50, confidence_threshold=0.7)
print(len(results))
print(len(val_keys))
y_true, y_pred = buildY(results)
    

print('len(y_true)', len(y_true))
print('len(y_pred)', len(y_pred))
# print('y_true', y_true)
# print('y_pred', y_pred)
print('y_true', np.unique(y_true))
print('y_pred', np.unique(y_pred))


# In[15]:


# plt.figure()
cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes=['car', 'pedestrian', 'truck'],
                      title='Confusion matrix, without normalization')

plt.show()


# In[16]:


results, scores = detection_out(pred_prob, keep_top_k=50, confidence_threshold=0.7)
y_true, y_pred, y_scores = buildYWithProb(results, scores)

print('len(y_scores)', len(y_scores))
print('len(y_true)', len(y_true))
print('len(y_pred)', len(y_pred))
# print('y_true', y_true)
# print('y_pred', y_pred)
# print('y_scores', np.array(y_scores))
print('y_true', np.unique(y_true))
print('y_pred', np.unique(y_pred))


# In[18]:


cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes=['misclassified', 'car', 'pedestrian', 'truck'],
                      title='Confusion matrix, without normalization')

plt.show()


# In[19]:


plot_confusion_matrix(cnf_matrix, normalize=True, classes=['missclassified', 'car', 'pedestrian', 'truck'],
                      title='Confusion matrix, with normalization')

plt.show()


# # Precision, Recall and F1-Score

# In[20]:


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# In[21]:


y_true = []
y_pred = []

for i, k in enumerate(val_keys[:2]):
    y_onehot = gt[k][:, 4:]
    c = np.zeros((len(y_onehot), 1))
    y_onehot = np.concatenate((c, y_onehot), axis=1) # add 0 for background
    y = np.argmax(y_onehot, axis=1).tolist()
    
    pred = results[i]
    
    pred_length = len(pred)
    y_length = len(y)
    
    if pred_length > 0:
        pred = pred[:, 0].astype(int).tolist()
        
    if pred_length > y_length:
        z = np.zeros(pred_length, dtype=int)
        z[:y_length] = y
        y = z.tolist()
    elif pred_length < y_length:
        z = np.zeros(y_length, dtype=int)
        z[:pred_length] = pred
        pred = z

    y_true.append(y)
    y_pred.append(pred)


# In[22]:


mapk(y_true, y_pred, k=10)


# In[23]:


mapk(y_true, y_pred, k=5)


# In[24]:


results, scores = detection_out(pred_prob, keep_top_k=50, confidence_threshold=0.7)
y_true, y_pred, _ = buildYWithProb(results, scores)
precision_score(y_true, y_pred, average='micro')


# In[25]:


results, scores = detection_out(pred_prob, keep_top_k=50, confidence_threshold=0.7)
recall_score(y_true, y_pred, average='weighted')
# precision_recall_curve([1, 2, 1], [0, 0, 0], average='weighted') # take a look at this case


# In[26]:


precision = dict()
recall = dict()
average_precision = dict()

y_true_onehot = np_utils.to_categorical(y_true, NUM_CLASSES)

for i in range(NUM_CLASSES):
    precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i],
                                                        y_scores[:, i])
    average_precision[i] = average_precision_score(y_true_onehot[:, i], y_scores[:, i])

# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_onehot.ravel(), y_scores.ravel())
average_precision["micro"] = average_precision_score(y_true_onehot, y_scores,
                                                     average="micro")


# In[27]:


# Plot Precision-Recall curve
lw = 2
colors = itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.clf()
plt.plot(recall[0], precision[0], lw=lw, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
plt.legend(loc="lower left")
plt.show()

# Plot Precision-Recall curve for each class
plt.clf()
plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i, color in zip(range(NUM_CLASSES), colors):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()


# In[28]:


print(classification_report(y_true, y_pred, target_names=['misclassfied', 'car', 'pedestrian', 'truck']))


# In[29]:


y_true_onehot = np_utils.to_categorical(y_true, NUM_CLASSES)
roc_auc_score(y_true_onehot, y_scores)


# # Predictions

# In[29]:


def build_inputs(keys):
    inputs = []
    images = []
    path_prefix = 'resources/udacity-dataset/'
    for key in keys:
        img_path = path_prefix + key
        img = image.load_img(img_path, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(imread(img_path))
        inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    return inputs, images

def plot(inputs, images):
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.9]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
#             label_name = CLASSES[label - 1]
            label_name = label - 1
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

        plt.show()


# In[11]:


input_shape=(300, 300, 3)
NUM_CLASSES = 3
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights.48-1.13.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES, priors)

CLASSES = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES, priors)
# In[30]:


keys = ['object-detection-crowdai/1479505369437332886.jpg']
inputs, images = build_inputs(keys)

plot(inputs, images)


# In[31]:


keys = ['object-detection-crowdai/1479502217225257061.jpg', 'object-detection-crowdai/1479498371963069978.jpg', 
       'object-detection-crowdai/1479503036282378933.jpg', 'object-dataset/1478019952686311006.jpg', 'object-dataset/1478019953689774621.jpg',
       'object-dataset/1478019955185244088.jpg', 'object-dataset/1478019959681353555.jpg', 'object-dataset/1478901535246276321.jpg',
       'object-dataset/1478901536388465963.jpg', 'object-dataset/1478901532389636298.jpg']
inputs, images = build_inputs(keys)

plot(inputs, images)


# In[32]:


from random import shuffle
keys = sorted(gt.keys())
shuffle(keys)
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]


# In[33]:


inputs = []
images = []
keys = list(map(lambda i: val_keys[i], range(60, 70)))
print(list(keys))
inputs, images = build_inputs(keys)
plot(inputs, images)


# In[27]:


import matplotlib.image as mpimg

def plot_gt(images, gt=[]):
    for i, img in enumerate(images):

        for row in gt:
            xmin = int(round(row[0] * img.shape[1]))
            ymin = int(round(row[1] * img.shape[0]))
            xmax = int(round(row[2] * img.shape[1]))
            ymax = int(round(row[3] * img.shape[0]))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),(0, 255, 0), 3)

        plt.imshow(img / 255.)
        plt.show()

def plot_pred(inputs, images, gt=[]):
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)
    print(results)
    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.7]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        
        currentAxis = plt.gca()
#         print('len', len(pred), len(top_conf))
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = CLASSES[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
            
#         for row in gt:
#             xmin = int(round(row[0] * img.shape[1]))
#             ymin = int(round(row[1] * img.shape[0]))
#             xmax = int(round(row[2] * img.shape[1]))
#             ymax = int(round(row[3] * img.shape[0]))
#             cv2.rectangle(img, (xmin, ymin), (xmax, ymax),(0, 255, 0), 3)

        plt.imshow(img / 255.)
        plt.show()


# In[28]:


k = val_keys[:1][0]
inputs, images = build_inputs(val_keys[:1])
plot_pred(inputs, images, gt[k])


# In[29]:


keys = ['object-detection-crowdai/1479502217225257061.jpg']
inputs, images = build_inputs(keys)
plot_pred(inputs, images, gt[keys[0]])


# In[12]:


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
#     print('xA', xA)
#     print('yA', yA)
#     print('xB', xB)
#     print('yB', yB)
    epsilon = 1e-8
 
    # compute the area of intersection rectangle
    x = (xB - xA + epsilon)
    y = (yB - yA + epsilon)
    if x <= epsilon or y <= epsilon:
        interArea = 0
    else:
        interArea = x * y
#     print('interArea', interArea)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#     print('boxAArea', boxAArea)
#     print('boxBArea', boxBArea)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def denormalize(val, w, h):
    return (val[0]*w, val[1]*h, val[2]*w, val[3]*h)

boxA = (0, 0, 1, 1)
boxB = (0.5, 0.5, 1.5, 1.5)
iou(boxA, boxB)


# In[150]:


def measure(keys):
    # predict
    inputs, images = build_inputs(keys)
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds, keep_top_k=50, confidence_threshold=0.7)
    
    for i, k in enumerate(keys):
        TP = 0
        FP = 0
        FN = 0
        img = images[i]
        y_onehot = gt[k]
        print('key', k)
        print('y_onehot count', len(y_onehot))
        print('pred count', len(results[i]))

        if len(results[i]) == 0:
            pred_bb = []
            y_prob = []
            y_pred = []
        else:
            pred_bb = results[i][:, 2:]
            y_prob = results[i][:, 1]
            y_pred = results[i][:, 0].astype(int) - 1 # -1 because we ignore background class, so class 0 becomes car

        gt_bb = gt[k][:, :4]
        y_gt = np.argmax(gt[k][:, 4:], axis=1)

        FN += abs(len(y_pred) - len(y_gt))

        d = dict()
        for j in range(len(pred_bb)):
            prob = y_prob[j]
            label = CLASSES[y_pred[j]]
            pred_bb_dnorm = denormalize(pred_bb[j], img.shape[1], img.shape[0])

            for k in range(len(gt_bb)):
                gt_bb_dnorm = denormalize(gt_bb[k], img.shape[1], img.shape[0])
                iou_val = iou(gt_bb_dnorm, pred_bb_dnorm)
                if iou_val > 0:
                    #(j, k): j=prediction index, k=groundtruth index
                    d[(j, k)] = iou_val
                    
        d_sorted = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        gt_detected = set()
        pred_detected = set()

        for k, v in d_sorted:
            print(k, v)
        print('---------------------------')

        for k, v in d_sorted:
            pred_idx, gt_idx = k

            if gt_idx not in gt_detected:
                
                if pred_idx not in pred_detected:
                    print('y_gt', y_gt[gt_idx])
                    print('y_pred', y_pred[pred_idx])
                    
                    if y_gt[gt_idx] == y_pred[pred_idx]:
                        TP += 1
                    else:
                        FP += 1
                    print(gt_idx, 'is selected with pred', pred_idx)
                    gt_detected.add(gt_idx)
                    pred_detected.add(pred_idx)
            elif pred_idx not in pred_detected:
                FP += 1
                print('No pred', pred_idx, 'but groundtruth', gt_idx, 'is detected already', (k, v))
                    
                    
        print('TP:', TP)
        print('FP:', FP)
        print('FN:', FN)


# In[11]:


keys = ['object-dataset/1478895407021175056.jpg']
inputs, images = build_inputs(keys)
plot_gt(images, gt[keys[0]])


# In[ ]:


inputs, images = build_inputs(keys)
plot_pred(inputs, images, gt[keys[0]])


# In[153]:


measure(keys)


# In[154]:


keys = ['object-detection-crowdai/1479502217225257061.jpg']
inputs, images = build_inputs(keys)
plot_gt(images, gt[keys[0]])


# In[155]:


inputs, images = build_inputs(keys)
plot_pred(inputs, images, gt[keys[0]])


# In[156]:


measure(keys)


# In[131]:


keys = ['object-dataset/1478896783257516407.jpg']
inputs, images = build_inputs(keys)
plot_gt(images, gt[keys[0]])


# In[132]:


inputs, images = build_inputs(keys)
plot_pred(inputs, images, gt[keys[0]])


# In[133]:


measure(['object-dataset/1478896783257516407.jpg'])


# In[157]:


keys = ['object-detection-crowdai/1479505929977588157.jpg']
inputs, images = build_inputs(keys)
plot_gt(images, gt[keys[0]])


# In[158]:


inputs, images = build_inputs(keys)
plot_pred(inputs, images, gt[keys[0]])


# In[159]:


measure(keys)


# In[137]:


measure(np.array(val_keys)[range(40, 50)])


# In[138]:


keys = ['object-dataset/1478021874581175763.jpg']
inputs, images = build_inputs(keys)
plot_gt(images, gt[keys[0]])


# In[139]:


inputs, images = build_inputs(keys)
plot_pred(inputs, images, gt[keys[0]])


# In[140]:


measure(keys)


# In[89]:





# In[13]:


iou((1551.7339324951172, 744.94636058807373, 1781.6526031494141, 1064.3930196762085), (504.0, 569.0, 561.0, 627.0))


# In[ ]:




