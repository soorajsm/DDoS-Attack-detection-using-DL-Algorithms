import os

from ryu.controller import ofp_event
from ryu.controller.handler import DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime

import numpy as np
import pandas as pd
import switch
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end-start))

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        file0 = open("PredictFlowStatsfile.csv", "w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        body = ev.msg.body
        for stat in sorted([flow for flow in body if flow.priority == 1], key=lambda flow: (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            if stat.match['ip_proto'] == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']
            else:
                icmp_code = -1
                icmp_type = -1
            tp_src = stat.match.get('tcp_src', 0) or stat.match.get('udp_src', 0)
            tp_dst = stat.match.get('tcp_dst', 0) or stat.match.get('udp_dst', 0)
            flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"
            try:
                packet_count_per_second = stat.packet_count / stat.duration_sec
                packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
            except ZeroDivisionError:
                packet_count_per_second = 0
                packet_count_per_nsecond = 0
            try:
                byte_count_per_second = stat.byte_count / stat.duration_sec
                byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
            except ZeroDivisionError:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0
            file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{stat.match['ip_proto']},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},{packet_count_per_second},{packet_count_per_nsecond},{byte_count_per_second},{byte_count_per_nsecond}\n")
        file0.close()




    def flow_training(self):
        if os.path.exists('./FCNN_TrainedModel.h5'):
            # Load the model if it exists
            print('-----------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('The Trained model is already present.')
            print('-----------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('Loading the model.........')
            model = load_model('./FCNN_TrainedModel.h5')

            self.flow_model = model  # Assign the trained model to flow_model

            flow_dataset = pd.read_csv('FlowStatsfile.csv')
            flow_dataset = flow_dataset.sample(frac=1).reset_index(drop=True)
            flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
            flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
            flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

            X_flow = flow_dataset.iloc[:, :-1].values
            X_flow = StandardScaler().fit_transform(X_flow)
            y_flow = flow_dataset.iloc[:, -1].values
            X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)


            scaler = StandardScaler()
            X_flow_train = scaler.fit_transform(X_flow_train)
            X_flow_test = scaler.transform(X_flow_test)

            y_flow_pre = model.predict(X_flow_test)
            y_flow_pred = (y_flow_pre > 0.5).astype(int)
            self.logger.info("------------------------------------------------------------------------------")
            self.logger.info("confusion matrix")
            cm = confusion_matrix(y_flow_test, y_flow_pred)
            self.logger.info(cm)
            acc = accuracy_score(y_flow_test, y_flow_pred)
            self.logger.info("success accuracy = {0:.2f} %".format(acc*100))
            fail = 1.0 - acc
            self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
            self.logger.info("------------------------------------------------------------------------------")

        else:
            print('-----------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('The Trained model not present.')
            print('-----------------------------------------------------------------------------------------------------------------------------------------------------------')
            self.logger.info("Flow Training ...")

            flow_dataset = pd.read_csv('FlowStatsfile.csv')
            flow_dataset = flow_dataset.sample(frac=1).reset_index(drop=True)
            flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
            flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
            flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

            X_flow = flow_dataset.iloc[:, :-1].values
            X_flow = StandardScaler().fit_transform(X_flow)
            y_flow = flow_dataset.iloc[:, -1].values

            # Undersample the majority class (label 1) before splitting
            count_class_0, count_class_1 = np.bincount(y_flow)
            indices_class_0 = np.where(y_flow == 0)[0]
            indices_class_1 = np.where(y_flow == 1)[0][:count_class_0]
            indices_undersampled = np.concatenate([indices_class_0, indices_class_1])
            X_flow_undersampled, y_flow_undersampled = X_flow[indices_undersampled], y_flow[indices_undersampled]

            # Shuffle the undersampled data
            X_flow_undersampled, y_flow_undersampled = shuffle(X_flow_undersampled, y_flow_undersampled)

            X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow_undersampled, y_flow_undersampled, test_size=0.15, random_state=0)


            model = Sequential([

                Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.07), input_shape=(X_flow_train.shape[1],)),Dropout(0.6),#Increase neurons in the first layer
                Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.07)),Dropout(0.65),  # Increase neurons in the second layer
                Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.07)),Dropout(0.65),  # Increase neurons in the third layer
                Dense(1, activation='sigmoid')  # Increase neurons in the last layer to 1
            ])


            # Compile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Train the model
            model.fit(X_flow_train, y_flow_train, epochs=5, batch_size=128, validation_data=(X_flow_test, y_flow_test))

            # Saving the model as a file in the current directory
            model.save('./FCNN_TrainedModel.h5')

            self.flow_model = model


            scaler = StandardScaler()
            X_flow_train = scaler.fit_transform(X_flow_train)
            X_flow_test = scaler.transform(X_flow_test)

            y_flow_pre = model.predict(X_flow_test)
            y_flow_pred = (y_flow_pre > 0.5).astype(int)
            self.logger.info("------------------------------------------------------------------------------")
            self.logger.info("confusion matrix")
            cm = confusion_matrix(y_flow_test, y_flow_pred)
            self.logger.info(cm)
            acc = accuracy_score(y_flow_test, y_flow_pred)
            self.logger.info("success accuracy = {0:.2f} %".format(acc*100))
            fail = 1.0 - acc
            self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
            self.logger.info("------------------------------------------------------------------------------")
            

            
            
    def flow_predict(self):
        try:
            file0 = open("predict.csv", "a")
            file0.write('label\n')
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')
            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')
            
            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = StandardScaler().fit_transform(X_predict_flow)
            y_flow_pred = self.flow_model.predict(X_predict_flow)
            
            # Convert predicted probabilities to integers (0 or 1)
            y_flow_pred_int = [int(round(pred[0])) for pred in y_flow_pred]

            file0.write("\n".join(map(str, y_flow_pred_int)))
            file0.close()
            legitimate_traffic = sum(1 for i in y_flow_pred_int if i == 0)
            ddos_traffic = sum(1 for i in y_flow_pred_int if i == 1)
            victim = int(predict_flow_dataset.iloc[y_flow_pred.argmax(), 5]) % 20
            self.logger.info("------------------------------------------------------------------------------")



            if ddos_traffic>legitimate_traffic:
                self.logger.info("ddos traffic ...")
                self.logger.info("victim is host: h{}".format(victim))
            else:
                self.logger.info("legitimate traffic ...")
                self.logger.info("------------------------------------------------------------------------------")


            # if legitimate_traffic / len(y_flow_pred) * 100 > 60:
            #     self.logger.info("legitimate traffic ...")
            #     self.logger.info("------------------------------------------------------------------------------")
                
            # else:
            #     self.logger.info("ddos traffic ...")
            #     self.logger.info("victim is host: h{}".format(victim))
            
                
        except Exception as e:
            self.logger.error("Error during flow prediction: {}".format(e))
