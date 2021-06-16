import os,sys

PSSE_PATH = r'C:\Program Files (x86)\PTI\PSSE33\PSSBIN'
PSSEVERSION = 33

if PSSEVERSION==34:
   import psse34               # it sets new path for psspy
else:
   sys.path.append(PSSE_PATH)

sys.path.append(PSSE_PATH)
os.environ['PATH'] = os.environ['PATH'] + ';' +  PSSE_PATH

import psspy

#--------------------------------
# PSS/E Saved case

CASE = r"C:\Program Files\PTI\PSSE32\EXAMPLE\savnw.sav"

def_char = psspy.getdefaultchar()
def_int = psspy.getdefaultint()
def_real = psspy.getdefaultreal()

psspy.psseinit(2000)
ierr_case_new = psspy.newcase_2([1,0],100.0,60,'','')
for bus_number in range(1,4):
    ierr_new_bus = psspy.bus_data_3(bus_number,[bus_number,1,1,1],[66,1.0,0.0,1.1,0.9,1.1,0.9],'')

ierr_new_line = psspy.branch_data(1,2,'1',[1,1,1,0,0,0],[0.03,0.08].extend([def_real for _ in range(13)]))
ierr_new_line = psspy.branch_data(1,3,'2',[1,1,1,0,0,0],[0.05,0.09].extend([def_real for _ in range(13)]))
ierr_new_line = psspy.branch_data(3,2,'3',[1,3,1,0,0,0],[0.03,0.07].extend([def_real for _ in range(13)]))

ierr_new_load = psspy.load_data_4(1,'1',[1,1,1,1,1,1],[200,80,0.0,0.0,0.0,0.0])


for machine_number in range(2,4):
    ierr_new_plant = psspy.plant_data(machine_number,[0],[1.0,100.0])
    ierr_new_machine = psspy.induction_machine_data(machine_number,'1', [def_int for _ in range(9)], [100,66,40].extend([def_real for _ in range(20)]))


ierr_fnsl = psspy.fnsl(options1=0, options5=0)
a = 0