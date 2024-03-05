import csv
import numpy as np
import pandas as pd
import math

class single_env():
    def __init__(self):
        self.action_space = 4
        self.observation_space = 9
        self.reward_origin = -float('inf')
        self.beta1 = 1
        self.beta2 = -0.1
        self.observation =[]
        self.observation_ =[]
        
        self.veh_DataRate = None
        self.veh_direction = None
        self.veh_packet_size = None
        self.beta = None

        
        self.max_delay = 0.58 #封包帶20,data_rate帶35
        self.min_delay = 0.14 #封包帶15,data_rate帶100
        
        
        self.max_data_rate = 100
        self.price = 0
        
        self.c = 1 #money_token
        self.rd = 0.1655 #每1Mbit的功耗
        self.rt = 0.7438 #每個時間段的功耗
        self.max_power_consumption = 18
        self.min_power_consumption = 6.5
        
    def minmax_norm(self,data, max_data, min_data):
        norm_data = (data-min_data) / (max_data-min_data)
        return norm_data
    
    def action_correspond(self, action, step_count): #動作對應,選擇出來的動作對應哪台車輛,先假設只有3台
        #print('step_count:',step_count)
        with open('simulation1_singlehop.csv','r',newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
        float_row = list(map(self.convert, rows[step_count])) #去讀第step_count行
        #print(float_row)
        if action == 0: #如果選出來的action為0,代表選擇K1車做傳輸
            obs_DataRate = float_row[0]
            direction = float_row[1]
            packet_size = float_row[8]
            
            
        elif action == 1 : #如果選出來的action為1,代表選擇K2車做傳輸
            obs_DataRate = float_row[2]
            direction = float_row[3]
            packet_size = float_row[8]
            
            
        elif action == 2: #如果選出來的action為2,代表選擇K3車做傳輸
            obs_DataRate = float_row[4]
            direction = float_row[5]
            packet_size = float_row[8]
        
        elif action == 3: #如果選出來的action為3,代表選擇MEC做傳輸
            obs_DataRate = float_row[6]
            direction = float_row[7]
            packet_size = float_row[8]
            
        
        return obs_DataRate, packet_size, direction
    
    def convert(self,string): #文字轉換成float
        try:
            string=float(string)
        except :
            pass    
        return string
    
    def Game_price_set(self, data_rate): #賽局的價格制定,依照bitrate給定
        if (35 <= data_rate < 50 ) :
            price = 1
        elif (50 <= data_rate <65 ) :
            price = 2
        elif (65 <= data_rate < 85) :
            price = 3
        elif (85 <= data_rate < 100) :
            price = 4
        elif (100 <= data_rate ) :
            price = 5
        else:
            price = 0
            print("price error")
        return price
    
    def choose_DataRate(self, DataRate_limit):
        if DataRate_limit > 100:
            DataRate_limit = 101 #這樣100才會被抓到
        data_rate = np.random.uniform(35,DataRate_limit)

        return data_rate
    
    def data_rate_norm(self, data_rate):
        norm_data_rate = data_rate / self.max_data_rate
        return norm_data_rate
    
    def reset(self):#讀取csv第一行
        with open('simulation1_singlehop.csv','r',newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
        
        float_row = list(map(self.convert, rows[1])) #第0列是title,用map直接全部轉換
        self.observation = float_row 
        ''' 另一種list文字轉換float的方法
        float_row = []
        for obj in rows[1]:
            float_row.append(self.convert(obj))
        print(float_row)
        '''
        #print(self.observation)
        return self.observation
        
    def step(self, action, nextstep_count):#要回傳(下一個狀態,獎勵,done,info)
        #寫done
        terminate = False #設定done
        if nextstep_count == 100: #因為我csv的資料只有100筆,所以當跑到第100筆時就讓done維true結束迴圈
            terminate = True #所以csv裡最後一筆資料訓練不到
        
        #寫observation_ 下一個狀態
        with open('simulation1_singlehop.csv','r',newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
        float_row = list(map(self.convert, rows[nextstep_count])) #去讀第nextstep_count行
        self.observation_ = float_row #為下一個狀態
        
        #動作進行對應
        self.veh_DataRate, self.veh_packet_size, self.veh_direction  = self.action_correspond(action, nextstep_count-1) #因為我設的nextstep_count是記錄下一個狀態,要找這個狀態的對應要nextstep_count-1
        #print('self.veh_throughput:',self.veh_throughput)
        #print('self.veh_direction:',self.veh_direction)
        #print('self.veh_packet_size:',self.veh_packet_size)
        self.beta = self.beta1 #因為這邊是單跳點所以這邊self.beta = 1
        #print('self.beta:',self.beta)
        
        #寫reward
        #計算車載傳輸延遲分數
        #bitrate = np.random.randint(1,self.veh_throughput +1 )#np.random.randint(a,b) 的範圍為a ~ b-1,所以self.veh_throughput 要+1
        data_rate = self.choose_DataRate(self.veh_DataRate)

        #print('data_rate:',data_rate)
        #print('self.veh_packet_size:',self.veh_packet_size)
        Transmission_delay = self.veh_packet_size / data_rate #計算車載傳輸延遲
        #print('Transmission_delay:',Transmission_delay)
        norm_Transmission_delay = self.minmax_norm(Transmission_delay, self.max_delay, self.min_delay) #做min_max正歸化
        #print('norm_Transmission_delay:',norm_Transmission_delay)
        
        #計算bitrate分數
        self.price = self.Game_price_set(data_rate)
        #print('price:',self.price)
        norm_data_rate = self.data_rate_norm(data_rate)
        data_rate_score = self.price * norm_data_rate
        #print('bitrate_score:',bitrate_score)
        
        #計算功耗分數,pig注意 考慮一下這段要不要做正規化,做正規化的話bitrate怎麼選利潤都會大於成本,不做正規化的話bitrate要選3以上利潤才會大於成本
        power_consumption = self.rd * data_rate + self.rt
        #print('power_consumption:',power_consumption)
        norm_power_consumption = self.minmax_norm(power_consumption, self.max_power_consumption, self.min_power_consumption)
        #print('norm_power_consumption:',norm_power_consumption)
        
        #計算reward
        reward = self.beta*(-norm_Transmission_delay + data_rate_score - (self.c * norm_power_consumption))
        #print('reward:',reward)
        return self.observation_, reward, terminate, {} #要回傳(下一個狀態,獎勵,done,info)