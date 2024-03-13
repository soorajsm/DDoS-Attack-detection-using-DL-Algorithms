from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from sklearn.preprocessing import StandardScaler

import switch
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.flow_model = None  # Initialize flow_model attribute

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end - start))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
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


    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)

    
    def _flow_stats_reply_handler(self, ev):

        timestamp = datetime.now()
        timestamp = timestamp.timestamp()

        file0 = open("PredictFlowStatsfile.csv", "w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')

        body = ev.msg.body
        icmp_code = -1
        icmp_type = -1
        tp_src = 0  # Initialize tp_src with a default value
        tp_dst = 0

        for stat in sorted([flow for flow in body if (flow.priority == 1)], key=lambda flow: (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):

            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']

            if stat.match['ip_proto'] == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']

            elif stat.match['ip_proto'] == 6:
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']

            elif stat.match['ip_proto'] == 17:
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']

            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)

            try:
                packet_count_per_second = stat.packet_count / stat.duration_sec
                packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
            except:
                packet_count_per_second = 0
                packet_count_per_nsecond = 0

            try:
                byte_count_per_second = stat.byte_count / stat.duration_sec
                byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
            except:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0

            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                        .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src, ip_dst, tp_dst,
                                stat.match['ip_proto'], icmp_code, icmp_type,
                                stat.duration_sec, stat.duration_nsec,
                                stat.idle_timeout, stat.hard_timeout,
                                stat.flags, stat.packet_count, stat.byte_count,
                                packet_count_per_second, packet_count_per_nsecond,
                                byte_count_per_second, byte_count_per_nsecond))

        file0.close()


    
        # self.logger.info("Flow Training ...")
        # flow_dataset = pd.read_csv('FlowStatsfile.csv')
        # flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
        # flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
        # flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')
        # X_flow = flow_dataset.iloc[:, :-1].values
        # X_flow = X_flow.astype('float64')
        # y_flow = flow_dataset.iloc[:, -1].values
        # X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)
        
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_flow_train.shape[1],)),
        #     tf.keras.layers.Dense(32, activation='relu'),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ])
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # self.flow_model = model.fit(X_flow_train, y_flow_train, epochs=1, batch_size=32, validation_split=0.2)

    def flow_training(self):

        self.logger.info("Flow Training ...")
    
        flow_dataset = pd.read_csv('FlowStatsfile.csv')
        flow_dataset = flow_dataset.drop(['idle_timeout', 'hard_timeout', 'flags'], axis=1)
        flow_dataset = flow_dataset.sample(frac=1).reset_index(drop=True)
        flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

        # X_flow = flow_dataset.iloc[:, :-1].values
        # X_flow = X_flow.astype('float64')
        # y_flow = flow_dataset.iloc[:, -1].values

        X = flow_dataset.drop('label',axis=1)
        y = flow_dataset['label']


        # X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25,
        #                                                                         random_state=0)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



        X_train=X_train.drop(['icmp_type', 'byte_count_per_nsecond', 'packet_count_per_second', 'byte_count', 'byte_count_per_second'],axis=1)
        X_test=X_test.drop(['icmp_type', 'byte_count_per_nsecond', 'packet_count_per_second', 'byte_count', 'byte_count_per_second'],axis=1)

        # X_test.drop(['icmp_type', 'byte_count_per_nsecond', 'packet_count_per_second', 'byte_count', 'byte_count_per_second','idle_timeout', 'hard_timeout', 'flags'],axis=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)



        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

        self.flow_model = model  # Assign the trained model to flow_model


        y_flow_pre = model.predict(X_test)
        y_flow_pred = (y_flow_pre > 0.5).astype(int)
        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info("confusion matrix")
        cm = confusion_matrix(y_test, y_flow_pred)
        self.logger.info(cm)
        acc = accuracy_score(y_test, y_flow_pred)
        self.logger.info("success accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            file0 = open("predict.csv", "w")
            file0.write('label\n')
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')
            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            predict_flow_dataset = predict_flow_dataset.drop(['icmp_type', 'byte_count_per_nsecond', 'packet_count_per_second', 'byte_count', 'byte_count_per_second','idle_timeout', 'hard_timeout', 'flags'],axis=1)

            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = X_predict_flow.astype('float64')
            
            if len(X_predict_flow) == 0:
                self.logger.info("No flow data to predict")
                return

            y_flow_pre = self.flow_model.predict(X_predict_flow)
            y_flow_pred = (y_flow_pre > 0.5).astype(int)



            file0.write("{}\n".format(y_flow_pred))

            file0.close()



            legitimate_trafic = (y_flow_pred == 0).sum()
            ddos_trafic = (y_flow_pred == 1).sum()

            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_trafic / len(y_flow_pred) * 100) > 80:
                self.logger.info("legitimate traffic ...")
            elif(ddos_trafic / len(y_flow_pred) * 100) > 10:
                self.logger.info("ddos traffic ...")
                # This line assumes victim index calculation is correct
                victim = int(predict_flow_dataset.iloc[y_flow_pred.argmax(), 5]) % 20
                self.logger.info("victim is host: h{}".format(victim))
            self.logger.info("------------------------------------------------------------------------------")
        except Exception as e:
            self.logger.error("Error in flow prediction: {}".format(e))
            pass
