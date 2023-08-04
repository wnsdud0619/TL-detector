#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
import message_filters
import scipy as scp
from utils import *
from model import *
import argparse
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from autoware_perception_msgs.msg import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_config():
    parser = argparse.ArgumentParser(description='DGIST-Trafficlight TensorFlow r1.14 implementation.')
    parser.add_argument('--checkpoint_path', type=str, default='/home/dgist/catkin_ws/src/dgist_trafficlight/src/model/model-20000')
    parser.add_argument('--legacy', action='store_true', help='Use existing topic name')
    parser.add_argument('--display', action='store_true', help='display processing time')
    args = parser.parse_args()
    return args

def main():
    rospy.init_node('inference', anonymous=True)
    args = parse_config()  # arg load

    input_size = (128, 256)  # h, w
    inference = TrafficlightInference(input_size, args.checkpoint_path, args.legacy, args.display)
    rospy.spin()


class TrafficlightInference(object):
    def __init__(self, input_size, check_point, use_legacy=False, use_display=False):
        self.input_size = input_size
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.tl_cls_to_idx = {'Unknown': 0, 'Green': 1, 'Red': 2, 'Yellow': 3, 'Green_Left': 4, 'Red_Left': 5}
        self.tl_colors = {'Unknown': (0, 0, 0), 'Green': (0, 255, 0), 'Red': (0, 0, 255), 'Yellow': (0, 255, 255), 'Green_Left': (255, 255, 0), 'Red_Left': (255, 0, 255)}
        self.tl_idx_to_cls = convert_str_to_index(self.tl_cls_to_idx)
        self.threshold = 0.9
        self.sess, self.model_op, self.image_pl = self.init_model(check_point)
        self.display = use_display

        TL_ALIVE_TOPIC = '/dgist/system/status/camera/traffic_light'
        TL_IMAGE_TOPIC = '/sensing/camera/traffic_light/image_raw/rois'
        TL_STATES_TOPIC = '/perception/camera/traffic_light_states'
        TL_ROI_TOPIC    = '/traffic_light_recognition/rois'
        if use_legacy:
            TL_STATES_TOPIC = '/perception/traffic_light_recognition/traffic_light_states'
            TL_ROI_TOPIC    = '/perception/traffic_light_recognition/rois'

        self.alive_msg = Bool()
        self.alive_msg.data = True
        self.alive_pub = rospy.Publisher(TL_ALIVE_TOPIC, Bool, queue_size=2)
        self.state_pub = rospy.Publisher(TL_STATES_TOPIC, TrafficLightStateArray, queue_size=2)
        sub_img_ = message_filters.Subscriber(TL_IMAGE_TOPIC, Image, queue_size=2)
        sub_roi_ = message_filters.Subscriber(TL_ROI_TOPIC, TrafficLightRoiArray, queue_size=2)
        timesync_sub = message_filters.TimeSynchronizer([sub_img_, sub_roi_], 2)
        timesync_sub.registerCallback(self.callback)

        print("=" * 80)
        print("Use legacy topic name = %s" % use_legacy)
        print("Display processing time = %s" % use_display)
        print("Model file = %s" % check_point)
        print("TL Image topic = %s" % TL_IMAGE_TOPIC)
        print("TL states topic = %s" % TL_STATES_TOPIC)
        print("TL RoIs topic = %s" % TL_ROI_TOPIC)
        print("=" * 80)

    def init_model(self, check_point, max_batch=3):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device('/gpu:0'):
                    image_pl = tf.placeholder(tf.float32, shape=[None, None, None, 3])
                    model = MultinetSeed(sess, image_pl)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            load_ckpt(sess, tf.get_collection(tf.GraphKeys.VARIABLES), check_point)
            model_op = model.out['pred_class']

            print('[Weight restored.]') 

            imgs = []
            print('Inference testing....') 
            for idx in range(max_batch):
                img = np.zeros(self.input_size, dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                imgs.append(img)

                feeds = {image_pl: np.array(imgs)}
                outputs = sess.run(model_op, feed_dict=feeds)
                print('tested : %d / %d...' %(idx + 1, max_batch))
            print('[Inference has been tested.]') 
        return (sess, model_op, image_pl)

    def decode_trafficlight(self, soft_max, th=0.9):
        class_ids = soft_max.argmax(1)  # max conf를 갖는 class 선택
        confs = soft_max.max(1)  # max conf 선택
        class_ids[confs <= th] = self.tl_cls_to_idx['Unknown']  # 기준 threshold 이하인 경우, Unknown으로 mapping
        results = list(zip(class_ids, confs))  # batch 별 하나의 class와 conf pair로 mapping
        return results

    def draw_trafficlight(self, img, rois_msgs, parsed_results):
        for idx, roi in enumerate(_rois):
            id_and_score = parsed_results[idx]

            # Unknown인 경우, drawing 안함
            if id_and_score[0] == self.tl_cls_to_idx['Unknown']:
                continue

            # bbox 계산
            x1 = roi_msg.roi.x_offset
            y1 = roi_msg.roi.y_offset
            x2 = x1 + roi_msg.roi.width
            y2 = y1 + roi_msg.roi.height

            # class, conf display
            class_name = self.tl_idx_to_cls[id_and_score[0]]
            color = self.tl_colors[class_name]
            msg = '%s:%0.2f' % (class_name, id_and_score[1])
            cv2.putText(img, msg, (int(x1), int(y2+25)), self.font, 1, color, thickness=2)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)

        return img

    def convertToAAPtype(self, _tl_result, conf):
    # model - 1 = Green, 2 = Red, 3 = Yellow, 4 = Green left, 5 = Red left
    # AAP   - 0 = Unknown, 1 = Red, 2 = Green, 3 = Yellow, 4 = Left, 5 = Right, 6 = Up, 7 = Down
        if _tl_result == 1:
            return [LampState(2, conf)]
        elif _tl_result == 2:
            return [LampState(1, conf)]
        elif _tl_result == 3:
            return [LampState(3, conf)]
        elif _tl_result == 4:
            return [LampState(2, conf), LampState(4, conf)]
        elif _tl_result == 5:
            return [LampState(1, conf), LampState(4, conf)]

        return [LampState(0, conf)]

    def callback(self, img_msg, roi_msg):
        start_time = time.time()

        # batch inference
        if roi_msg.rois:
            np_img = np.fromstring(img_msg.data, np.uint8).reshape(img_msg.height, img_msg.width, 3)
            roi_imgs = np.array(np.split(np_img, len(roi_msg.rois), axis=1))
            
            feeds = {self.image_pl: roi_imgs}
            outputs = self.sess.run(self.model_op, feed_dict=feeds)
            parsed_results = self.decode_trafficlight(outputs, self.threshold)

        # 결과 packing
        TrafficLightStateArray_msg = TrafficLightStateArray()  # 한씬의 신호등
        max_conf_state = [LampState(-1, 0.0)]
        signal_idx = []
        for idx, roi in enumerate(roi_msg.rois):
            lamp_sate_msgs = self.convertToAAPtype(parsed_results[idx][0], parsed_results[idx][1])  # 각 램프 정의
            TrafficLightState_msg = TrafficLightState(id=roi.id, lamp_states=lamp_sate_msgs)  # 하나의 신호등 정의
            if parsed_results[idx][0] != 0:
                if max_conf_state[0].confidence < parsed_results[idx][1]:
                    max_conf_state = TrafficLightState_msg.lamp_states
                else:
                    signal_idx.append(idx)

            TrafficLightStateArray_msg.states.append(TrafficLightState_msg)  # 한씬의 신호등 정의
 
        # 한씬에서 max conf의 class로 통합
        if max_conf_state[0].type != -1:
            for idx in signal_idx:
                TrafficLightStateArray_msg.states[idx].lamp_states = max_conf_state
    
        TrafficLightStateArray_msg.header = roi_msg.header
        self.state_pub.publish(TrafficLightStateArray_msg)
        self.alive_pub.publish(self.alive_msg)

        if not self.display:
            return
        duration = (time.time() - start_time) * 1000
        print('%.1f ms(RoIs:%d)' % (duration, len(roi_msg.rois)))


if __name__ == '__main__':
    main()
