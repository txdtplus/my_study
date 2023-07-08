class Router:

   def __init__(self, name, mode, number):
       self.name = name
       self.mode = mode
       self.number = number
       self.l3protocol = 'arp, static, rip, eigrp, ospf, isis, bgp'

   def desc(self):
       print(f'This is {self.name}_{self.mode}_{self.number}router')
       print(f'l3func: {self.l3protocol}')


class Switch(Router):

   def __init__(self, name, mode, number, zone):
       super(Switch, self).__init__(name, mode, number) # 继承super class __init__属性
       self.zone = zone


if __name__ == '__main__':
   huawei = Switch('HUAWEI', 'CE', '12808', 'USA')
   huawei.l3protocol = 'css, vxlan, evpn, mbgp, srv6'
   huawei.desc()