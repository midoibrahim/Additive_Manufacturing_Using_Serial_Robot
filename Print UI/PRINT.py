import sys
import time
from printrun.pronterface import *  # Pronterface API
from robolink import *    # API to communicate with RoboDK
from robodk import *      # basic matrix operations
import re
import os
import wx
from threading import Thread
from datetime import datetime, timedelta
from viztracer import VizTracer
from multiprocessing import Process ,Queue
import requests

#---------------------------------WEBSITE-------------------------------------------------------------------------------------------------
from flask import Flask, render_template, Response, request , stream_with_context,redirect
import cv2
import os, sys
import numpy as np
from threading import Thread
import os
from werkzeug.utils import secure_filename
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import random
from typing import Iterator


#-----------------------------------------------
import pyrealsense2 as rs
import mediapipe as mp
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import matplotlib.pyplot as plt
import time
import math
from robolink import *    # RoboDK API
from robodk import *      # Robot toolbox
import tempfile


global capture,rec_frame, grey, switch, neg, face, rec, out,camera,app3
task=" "
webcontrol=0
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
file_list=[]
increment=0
camera=1
global lg
lg=0
spastop=0
ins_id=0
ins_count=1

def pause2():
    response = requests.get("http://localhost:5000/pausespa")
timer = threading.Timer(60,pause2)

for file in os.listdir("gcode"):
    if file.endswith(".gcode"):
        file_list.append(file)

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#inztatiaze flask app
app3 = Flask(__name__, template_folder='./templates')

# app3.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app3.config['UPLOAD_EXTENSIONS'] = ['.gcode']
global image



def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame,image,q
    # cap=cv2.VideoCapture(0)
    while 1:
        # x,frame=cap.read()
        frame = q.get()
        # frame=image
        # cv2.imshow("frame",frame)
        if(capture):
            capture=0
            now = datetime.now()
            p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
            cv2.imwrite(p,cv2.flip(frame,1))

        if(rec):
            rec_frame=cv2.flip(frame,1)
            frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
            frame=cv2.flip(frame,1)
        if camera:
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

@app3.route('/')
def index():
    return render_template('index.html',error=task,files_obj = file_list,inc=increment)

def generate():
    global lg
    while lg <= 100:
        # lg=lg+1
        lg=round(lg,2)
        yield "data:" + str(lg) + "\n\n"
        time.sleep(1)
    yield "data:" + str(100) + "\n\n"

def generate2():
    global ins_id,ins_count
    # ins_id=0
    # ins_count=50
    while ins_id < ins_count:
        # ins_id=ins_id+1
        s=round(ins_id/ins_count*100,2)
        yield "data:" + str(s) + "\n\n"
        time.sleep(1)
    yield "data:" + str(100) + "\n\n"

def generate3():
    global task
    while 1:
        yield "data:" + str(task) + "\n\n"
        time.sleep(1)

@app3.route('/status')
def status():
	return Response(generate3(), mimetype= 'text/event-stream')

@app3.route('/progress')
def progress():
	return Response(generate(), mimetype= 'text/event-stream')

@app3.route('/progress2')
def progress2():
	return Response(generate2(), mimetype= 'text/event-stream')

@app3.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@app3.route("/chart-data")
def chart_data() -> Response:
    response = Response(stream_with_context(generate_random_data()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response

def generate_random_data() -> Iterator[str]:
    """
    Generates random value between 0 and 100
    :return: String containing current timestamp (YYYY-mm-dd HH:MM:SS) and randomly generated data.
    """
    global job
    try:
        while True:
            json_data = json.dumps(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    # "value": random.random() * 100,
                    "value":job.app.get_temp()[0]
                }
            )
            yield f"data:{json_data}\n\n"
            time.sleep(1)
    except GeneratorExit:
        pass


@app3.route('/selector', methods = ['GET', 'POST'])
def select_file():
    global file_list,gfile,app2,job,webcontrol,RunMode,runandload,flagdel
    if request.method == 'POST':
        gfile=request.form.get('gcodefile')
        mode=request.form.getlist('mode')
        print("Mode is"," ".join(mode))
        if 'runonrobot' in mode:
            RunMode=RUNMODE_RUN_ROBOT
        else:
            RunMode=RUNMODE_SIMULATE
        if 'runandload' in mode:
            runandload=1
        else:
            runandload=0
        task=str(gfile)+" has been selected \n"
        if app2.mainwindow.filename is None:
            app2.mainwindow.filename=os.path.join(current_directory,'gcode',gfile)
        print(app2.mainwindow.filename)
        webcontrol=1
    # return redirect('/')
    return render_template('index.html',error=task,files_obj = file_list,inc=increment)



@app3.route('/spa', methods = ['POST','GET'])
def spa():
    global timer
    if not timer.is_alive():
        timer = threading.Timer(60,pause2)
        timer.start()
        print("Modal timer started")
        return render_template('modal.html')
    return '',204

@app3.route('/pausespa', methods = ['POST','GET'])
def pausespa():
    global task,app2,file_list,increment,spastop
    if not spastop:
        task="Paused print (Sphagettie Detected)"
        app2.mainwindow.pause(toggle=1)
        # app2.mainwindow.do_settemp("0")
        # time.sleep(4)
        app2.mainwindow.p.send_now("G1 E-2")
        # home()
        print(task)
        spastop=1
    return '',204

@app3.route('/resumespa', methods = ['POST','GET'])
def resumespa():
    global task,app2,file_list,increment,timer
    timer.cancel()
    print("Modal timer canceled")
    task="Sphagettie FP-Continue"
    print(task)
    return '',204

@app3.route('/pause', methods = ['POST','GET'])
def pause():
    global task,app2,file_list,increment
    task="Paused print (Collision avoidance)"
    app2.mainwindow.pause(toggle=1)
    print("Paused by Collision avoidance")
    return '',204

@app3.route('/resume', methods = ['POST','GET'])
def resume():
    global task,app2,file_list,increment,spastop
    task="Resumed print"
    # try:
    if not spastop:
        app2.mainwindow.pause(toggle=2)
        print("Resumed by request")
    # except:
    #     pass
    return '',204

@app3.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    global file_list
    if request.method == 'POST':
        f = request.files['file']
        filename=secure_filename(f.filename)
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app3.config['UPLOAD_EXTENSIONS']:
            task="Must upload a Gcode extension"
        else:
            f.save(os.path.join("gcode",filename))
            task=str(filename)+" uploaded successfully"
            file_list.append(filename)
            file_list=np.unique(file_list).tolist()
    return render_template('index.html',error=task,files_obj = file_list,inc=increment)

@app3.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera,task,file_list,app2,increment,flagdel,webcontrol,runandload,RunMode
    webcontrol=1
    if request.method == 'POST':
        inc=str(request.form['inc'])
        increment = int(inc)
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
            task="Capture"
        elif  request.form.get('pause') == 'Pause':
            if not app2.mainwindow.p.printing:
                mode=request.form.getlist('mode')
                print("Mode is"," ".join(mode))
                if 'runonrobot' in mode:
                    RunMode=RUNMODE_RUN_ROBOT
                else:
                    RunMode=RUNMODE_SIMULATE
                if 'runandload' in mode:
                    runandload=1
                else:
                    runandload=0
                RunMode=RUNMODE_RUN_ROBOT
                app2.mainwindow.printfile()
                task="Started printing"
                return redirect('/')
            else:
                global spastop
                if spastop:
                    spastop=0
                    app2.mainwindow.p.send_now("G1 E2")
                app2.mainwindow.pause()

        elif  request.form.get('home') == 'Home':
            task="Homing"
            home()
        elif  request.form.get('delete') == 'Delete':
            task="Delete output"
            app2.mainwindow.paused=True
            app2.mainwindow.p.printing=False
            job.deleteout()
            job.StartPrint()
        elif  request.form.get('upx') == 'Upx':
            task="+X "+str(increment)+" mm"
            moveXYZ(x=increment)
        elif  request.form.get('upy') == 'Upy':
            task=("+Y ")+str(increment)+" mm"
            moveXYZ(y=increment)
        elif  request.form.get('upz') == 'Upz':
            task=("+Z ")+str(increment)+" mm"
            moveXYZ(z=increment)
        elif  request.form.get('downx') == 'Downx':
            task=("-X ")+str(increment)+" mm"
            moveXYZ(x=(increment*-1))
        elif  request.form.get('downy') == 'Downy':
            task=("-Y ")+str(increment)+" mm"
            moveXYZ(y=(increment*-1))
        elif  request.form.get('downz') == 'Downz':
            task=("-Z ")+str(increment*-1)+" mm"
            moveXYZ(z=(increment*1))
        elif  request.form.get('stop') == 'Stop/Start':
            if(switch==1):
                switch=0
                camera= 0
                cv2.destroyAllWindows()
                task="Stop surveillance"
            else:
                camera = 1
                switch=1
                task="Resume surveillance"
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            task="Recording"
            rec= not rec
            if(rec):
                now=datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                path="video/vid_{}.avi".format(str(now).replace(":",'').replace(" ","_"))
                print(path)
                out = cv2.VideoWriter(path, fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
    # elif request.method=='GET':
    #     return render_template('index.html',error=task,files_obj = file_list,inc=increment)
    # return render_template('index.html',error=task,files_obj = file_list,inc=increment)
    return '',204

def Run_Web():
    app3.run(debug=False ,threaded=True,host="0.0.0.0")

#--------------------------------------------------------------------------------------------------------------------------------------


RunMode=RUNMODE_RUN_ROBOT
#RUNMODE_RUN_ROBOT, RUNMODE_SIMULATE

tracer = VizTracer(min_duration=200)
tracer.start()
runandload=0
current_directory= str(os.path.dirname(os.path.realpath(__file__)))

# Python program using traces to kill threads
class thread_with_trace(threading.Thread):
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, event, arg):
        if event == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        if self.killed:
            if event == 'line':
                print("Killing")
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True


def moveXYZ(x=0,y=0,z=0):
    try:
        target = robot2.Pose()               # retrieve the current target as a pose (position of the active tool with respect to the active reference frame)
        xyzabc = Pose_2_KUKA(target)        # Convert the 4x4 pose matrix to XYZABC position and orientation angles (mm and deg)
        xi,yi,zi,ai,bi,ci = xyzabc                # Calculate a new pose based on the previous pose
        xyzabc2 = [xi+x,yi+y,zi+z,ai,bi,ci]
        target2 = KUKA_2_Pose(xyzabc2)      # Convert the XYZABC array to a pose (4x4 matrix)
        robot2.MoveJ(target2)                # Make a linear move to the calculated position
        print("Moving Relative(",x,",",y,",",z,")mm")
    except:
        print("Traget cannot be reached")
        pass
def speedchange():
        global F_Value
        robot2.setSpeed((F_Value/60))

def home():
    try:
        robot.MoveJ([0,0,0,0,0,0])
    except:
        print("Homing cannot be reached")
        pass

class MyWindow(PronterWindow):
    def printfile(self, event=None):
        global job,t2
        self.p.printing=True
        self.on_startprint()
        t2 = thread_with_trace(name="Robodk", target=job.Run)
        t2.setDaemon(True)
        t2.start()

    def pause(self, event = None,toggle=0):
        global task
        if toggle==0:
            if not self.paused:
                self.log(_("Print paused at: %s") % format_time(time.time()))
                self.paused = True
                wx.CallAfter(self.pausebtn.SetLabel, _("Resume"))
                wx.CallAfter(self.toolbarsizer.Layout)
                task="Print paused at: "+str(format_time(time.time()))
            else:
                self.log(_("Print resumed at: %s") % format_time(time.time()))
                self.paused = False
                wx.CallAfter(self.pausebtn.SetLabel, _("&Pause"))
                wx.CallAfter(self.toolbarsizer.Layout)
                task="Print resumed at: "+str(format_time(time.time()))
        elif toggle==1:
            self.log(_("Print paused at: %s") % format_time(time.time()))
            self.paused = True
            wx.CallAfter(self.pausebtn.SetLabel, _("Resume"))
            wx.CallAfter(self.toolbarsizer.Layout)
            # task="Print paused at: "+str(format_time(time.time()))
        elif toggle==2:
            self.paused = False
            wx.CallAfter(self.pausebtn.SetLabel, _("&Pause"))
            wx.CallAfter(self.toolbarsizer.Layout)
            # task="Print resumed at: "+str(format_time(time.time()))


    def homeButtonClicked(self, axis):
        # When user clicks on the XY control, the Z control no longer gets spacebar/repeat signals
        self.zb.clearRepeat()
        home()

    def moveXY(self, x, y):
        # When user clicks on the XY control, the Z control no longer gets spacebar/repeat signals
        self.zb.clearRepeat()
        if x != 0 or y !=0:
            moveXYZ(x=x,y=y)

    def moveZ(self,z):
        if z != 0:
            moveXYZ(z=z)
        # When user clicks on the Z control, the XY control no longer gets spacebar/repeat signals
        self.xyb.clearRepeat()

    def do_setspeed(self, l = ""):
        try:
            if not isinstance(l, str) or not len(l):
                l = str(self.speed_slider.GetValue())
            else:
                l = l.lower()
            self.speed = int(l)/100
            if self.p.online:
                self.p.send_now("M220 S" + l)
                self.log(_("Setting print speed factor to %f%%.") % self.speed)
                speedchange()
            else:
                self.logError(_("Printer is not online."))
        except Exception as x:
            self.logError(_("You must enter a speed. (%s)") % (repr(x),))

    def get_time(self):
        gcode = self.fgcode
        return gcode.estimate_duration()

    def get_temp(self):
        try:
            temps = parse_temperature_report(self.tempreadings)
            if "T0" in temps and temps["T0"][0]:
                hotend_temp = float(temps["T0"][0])
                hotend_settemp = float(temps["T0"][1])
            elif "T" in temps and temps["T"][0]:
                hotend_temp = float(temps["T"][0])
                hotend_settemp = float(temps["T"][1])
            else:
                hotend_temp = 0
                hotend_settemp = 0
            return hotend_temp,hotend_settemp
        except:
            self.logError(traceback.format_exc())

class MyApp(PronterApp):
    mainwindow = None
    def __init__(self, *args, **kwargs):
        super(PronterApp, self).__init__(*args, **kwargs)
        self.SetAppName("Pronterface")
        self.locale = wx.Locale(wx.Locale.GetSystemLanguage())
        self.mainwindow = MyWindow(self)
        self.mainwindow.Show()



class ABBPrint():
    def __init__(self):
        global t2,app2
        time.sleep(5)
        print("Sim thread started")
        self.app = app2.mainwindow
        self.p = self.app.p
        self.app.speed=1


    def createprogram(self):
        g0pattern = r"^G0.*"
        g1pattern = r"^G1.*"
        g0re = re.compile(g0pattern)
        g1re = re.compile(g1pattern)

        Xpattern = r".*X[-\.\d]+"
        Ypattern = r".*Y[-\.\d]+"
        Zpattern = r".*Z[-\.\d]+"
        Fpattern = r".*F[-\.\d]+"
        Epattern = r".*E[-\.\d]+"
        Mpattern = r"^M.*"

        Xre = re.compile(Xpattern)
        Yre = re.compile(Ypattern)
        Zre = re.compile(Zpattern)
        Fre = re.compile(Fpattern)
        Ere = re.compile(Epattern)
        Mre = re.compile(Mpattern)

        currentX = 0
        currentY = 0
        currentZ = 0
        currentF = 0
        currentE = 0
        currentM = 0

        path_file = os.path.join(current_directory,'gcode/out.csv')
        print(path_file)
        fout = open(path_file, 'w')
        should_update = False
        if str(self.app.filename).split(".")[1] == 'stl':
            self.app.filename = str(self.app.filename).split(".")[0]+"_export.gcode"
            try:
                os.remove(self.app.filename)
            except:
                pass
        print("app filename is:" + self.app.filename)

        while not os.path.exists(self.app.filename):
            time.sleep(0.2)
            print(self.app.filename)
            os.path.exists(self.app.filename)

        with open(self.app.filename) as f:
            lines = f.readlines()
            for line in lines:
                if Mre.match(line):
                    currentM = line.split(';')[0].splitlines()[0]
                    should_update = True
                elif g0re.match(line):
                    if Zre.match(line):
                        should_update = True
                        nextZ = re.findall(Zpattern, line)[0].split('Z')[1]
                        if currentZ != nextZ:
                            currentZ = nextZ
                elif g1re.match(line):
                    if Xre.match(line):
                        should_update = True
                        nextX = re.findall(Xpattern, line)[0].split('X')[1]
                        if currentX != nextX:
                            currentX = nextX
                    if Yre.match(line):
                        should_update = True
                        nextY = re.findall(Ypattern, line)[0].split('Y')[1]
                        if currentY != nextY:
                            currentY = nextY
                    if Zre.match(line):
                        should_update = True
                        nextZ = re.findall(Zpattern, line)[0].split('Z')[1]
                        if currentZ != nextZ:
                            currentZ = nextZ
                    if Fre.match(line):
                        should_update = True
                        nextF = re.findall(Fpattern, line)[0].split('F')[1]
                        if currentF != nextF:
                            currentF = nextF
                    if Ere.match(line):
                        should_update = True
                        nextE = re.findall(Epattern, line)[0].split('E')[1]
                        if currentE != nextE:
                            currentE = nextE
                    else:
                        currentE = 0
                if should_update:
                    fout.write("{0},{1},{2},{3},{4},{5}\n".format(
                        currentX, currentY, currentZ, currentF, currentE, currentM))
                    should_update = False
                    currentM=0

        fout.close()
        global program
        # Set the name of the reference frame to place the targets:
        REFERENCE_NAME = 'Print'

        # Set the name of the reference target
        # (orientation will be maintained constant with respect to this target)
        TARGET_NAME = 'Home'

        # ---------------------------
        # Start the RoboDK API
        RDK = Robolink()

        # Ask the user to pick a file:
        #rdk_file_path = RDK.getParam("PATH_OPENSTATION")
        #path_file = 'F:/Mohamed/Desktop/out.csv'
        if not path_file:
            print("Nothing selected")
            quit()

        # Get the program name from the file path
        program_name = getFileName(path_file)

        # Load the CSV file as a list of list [[x,y,z,speed],[x,y,z,speed],...]
        data = LoadList(path_file)

        # Delete previously generated programs that follow a specific naming
        # Automatically delete previously generated items (Auto tag)
        list_items = RDK.ItemList()  # list all names
        for item in list_items:
            if item.Name().startswith('out'):
                item.Delete()

        # Select the robot (the popup is diplayed only if there are 2 or more robots)
        robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
        if not robot.Valid():
            raise Exception("Robot not selected or not valid")
            quit()

        # Get the reference frame to generate the path
        frame = RDK.Item(REFERENCE_NAME, ITEM_TYPE_FRAME)
        if not frame.Valid():
            raise Exception("Reference frame not found. Use name: %s" % REFERENCE_NAME)

        # Use the home target as a reference
        target = RDK.Item(TARGET_NAME, ITEM_TYPE_TARGET)
        if not target.Valid():
            raise Exception("Home target is not valid. Set a home target named: %s" % TARGET_NAME)

        # Set the robot to the home position
        robot.setJoints(target.Joints())

        # Get the pose reference from the home target
        pose_ref = robot.Pose()

        # Add a new program
        program = RDK.AddProgram(program_name, robot)

        # Turn off rendering (faster)
        RDK.Render(False)

        # Speed up by not showing the instruction:
        program.ShowInstructions(False)

        # Remember the speed so that we don't set it with every instruction
        current_speed = None
        target = None

        # Very important: Make sure we set the reference frame and tool frame so that the robot is aware of it
        program.setPoseFrame(frame)
        program.setPoseTool(robot.PoseTool())
        program.setRounding(2)
        # Iterate through all the points
        for i in range(len(data)):
            pi = pose_ref
            pi.setPos(data[i])

            # Update speed if there is a 4th column
            if len(data[i]) >= 3:
                speed = data[i][3]/60
            # Update the program if the speed is different than the previously set speed
            if type(speed) != str and speed != current_speed:
                program.setSpeed(speed, accel_linear=1000)
                current_speed = speed

            if len(data[i]) >= 4:
                E = data[i][4]
            # Update the program if the speed is different than the previously set speed
            if E != 0:
                program.RunInstruction("Extruder " + str(E), 0)

            if len(data[i]) >= 5:
                M = data[i][5]
            if M != 0:
                program.RunInstruction("Mcode " + str(M), 0)

            target = RDK.AddTarget('T%i' % i, frame)
            target.setPose(pi)
            pi = target

            # Add a linear movement (with the exception of the first point which will be a joint movement)
            if i == 0:
                program.MoveJ(pi)
            else:
                program.MoveL(pi)
            global lg
            lg=100*i/len(data)
            if i % 400 == 0:
                program.ShowTargets(False)
                print("Loading %s: %.1f %%" %(program_name, 100*i/len(data)))
                RDK.Render(False)
        lg=100
        program.ShowTargets(False)
        RDK.Render(False)
        RDK.ShowMessage("Done", False)
        print("Done")

        check_collisions = COLLISION_OFF
        # Update the path (can take some time if collision checking is active)
        update_result = program.Update(check_collisions)
        # Retrieve the result
        n_insok = update_result[0]
        timemove = update_result[1]
        distance = update_result[2]
        percent_ok = update_result[3]*100
        str_problems = update_result[4]
        if percent_ok < 100.0:
            msg_str = "WARNING! Problems with <strong>%s</strong> (%.1f):<br>%s" % (program_name, percent_ok, str_problems)
        else:
            msg_str = "No problems found for program %s" % program_name

        # Notify the user:
        print(msg_str)
        print("Time Expected= "+str(timemove))
        print(msg_str)

    def preheat(self):
        self.app.do_settemp("200")
        self.p.send_now("M106 P1 S255")
        self.p.send_now("M106 P0 S100")
        self.p.send_now("G91")
        time.sleep(4)
        hotend_temp, hotend_settemp = self.app.get_temp()
        print("Preheating", hotend_temp, "/", hotend_settemp)
        while abs(hotend_settemp-hotend_temp) > 3:
            hotend_temp, hotend_settemp = self.app.get_temp()
            time.sleep(1)
        print("Preheating Done")

    def progresscalc(self):
        global start, dur, t2, t
        now = datetime.now()
        done = now-start
        done = done.total_seconds()
        progress = round(int(done)/int(dur)*100, 2)
        print("progress: ",progress,"%")
        if(t2.is_alive()):
            while(self.app.paused):
                time.sleep(1)
            tp = threading.Timer(300, self.progresscalc)
            tp.setDaemon(True)
            tp.start()
        else:
            t.join()
            sys.exit()

    def deleteout(self):
        list_items = RDK.ItemList()  # list all names
        for item in list_items:
            if item.Name().startswith('out'):
                item.Delete()

    def StartPrint(self):
        global RDK, program, app2,t3,robot,robot2,t2,runandload,flag,task,app3
        RDK = Robolink()
        tool = RDK.ItemUserPick("Select a tool for incremental movements", ITEM_TYPE_TOOL)
        robot2 = tool.Parent()
        robot = RDK.Item('ABB IRB120')      # retrieve the robot by name
        self.preheat()
        flag=1
        list_items = RDK.ItemList()  # list all names
        for item in list_items:
            if item.Name().startswith('out'):
                flag=0
        if flag:
            while(self.app.filename == None):
                task="Upload Stl/Gcode"
                print("Upload Stl/Gcode")
                # robot2.Connect("192.168.125.1")
                #self.app.homeButtonClicked("all")
                #self.app.moveZ(-10)
                #self.app.moveXY(10,10)
                time.sleep(3)
            if not webcontrol:
                dlg = wx.MessageDialog(None, "Run and Load?", "Printing Mode",wx.YES_NO | wx.ICON_QUESTION)
                retCode = dlg.ShowModal()
                if (retCode == wx.ID_YES):
                    print("Run and Load Mode")
                    runandload=1
                else:
                    print("Load First Mode")
                    runandload=0
                dlg.Destroy()

            if(runandload):
                t3 = thread_with_trace(name='Create_Prog', target=self.createprogram)
                t3.start()
                print("Thread create began..wait for 20 secs")
                time.sleep(20)
                t2 = thread_with_trace(name="Robodk", target=self.Run)
                t2.setDaemon(True)
                t2.start()
            else:
                self.createprogram()
                if (webcontrol):
                    app2.mainwindow.printfile()
        else :
            wx.CallAfter(self.app.printbtn.SetLabel, _("Print"))
            wx.CallAfter(self.app.printbtn.Enable)
            print("Press Print Button To Start")

    def Run(self):
        global RDK, program, app2,t3,robot,F_Value,webcontrol,RunMode,ins_id,ins_count

        if not webcontrol:
            dlg = wx.MessageDialog(None, "Run On Robot?", "Choose Run Mode",wx.YES_NO | wx.ICON_QUESTION)
            retCode = dlg.ShowModal()
            if (retCode == wx.ID_YES):
                print("RUN ON ROBOT MODE ON")
                RunMode=RUNMODE_RUN_ROBOT
            else:
                print("SIMULATION MODE ON")
                RunMode=RUNMODE_SIMULATE
            dlg.Destroy()

        self.app.on_startprint()

        prog = RDK.Item('out', ITEM_TYPE_PROGRAM)
        print("Starting Printing")

        self.app.paused=0

        if RunMode == RUNMODE_RUN_ROBOT:
            state = robot.Connect("192.168.125.1")
            state, msg = robot.ConnectedState()
            print(state)
            print(msg)
        robot.setJoints([0, 0, 0, 0, 0, 0])      # set all robot axes to zero
        RDK.setRunMode(RunMode)
        print(RunMode)
        # Iterate through all the instructions in a program:
        ins_id = 0
        ins_count = prog.InstructionCount()
        print("Instruction count is",ins_count)
        global start, dur, t
        start = datetime.now()
        end = timedelta(seconds=prog.Update()[1])
        dur = end.total_seconds()
        # t = Thread(name="progress", target=self.progresscalc)
        # t.setDaemon(True)
        # t.start()
        robot.setRounding(2)

        while ins_id < ins_count:
            if not self.app.paused:
                try:
                    # Retrieve instruction
                    ins_nom, ins_type, move_type, isjointtarget, pose, joints = prog.Instruction(ins_id)

                    if ins_type == INS_TYPE_CHANGESPEED:
                        name = ins_nom
                        F_Value = float(re.findall("[+-]?\d+\.\d+", name)[0])
                        robot.setSpeed(F_Value*self.app.speed)  # Add the set speed instruction
                        F_Value *= 60
                        Command = 'G1 F'+str(int(F_Value))
                        if RunMode == RUNMODE_RUN_ROBOT:
                            while not self.p.online:
                                print("not online")
                                time.sleep(0.1)
                            self.p.send_now(Command)

                    elif ins_type == INS_TYPE_MOVE:
                        if move_type == MOVE_TYPE_JOINT:
                            if isjointtarget == 1:
                                robot.MoveJ(joints)
                            else:
                                robot.MoveJ(pose)

                        elif move_type == MOVE_TYPE_LINEAR:
                            if isjointtarget == 1:
                                robot.MoveL(joints)
                            else:
                                robot.MoveL(pose)

                    elif "Extruder" in ins_nom:
                        name = str(ins_nom)
                        E_Value = float(re.findall("[+-]?\d+\.\d+", name)[0])
                        Command = 'G1 E'+str(E_Value)
                        if RunMode == RUNMODE_RUN_ROBOT:
                            while not self.p.online:
                                print("not online")
                                time.sleep(0.1)
                            self.p.send_now(Command)

                    elif "Mcode" in ins_nom:
                        name = str(ins_nom)
                        M_Value = name.split(' ',1)[1]
                        Command = str(M_Value)
                        if RunMode == RUNMODE_RUN_ROBOT:
                            while not self.p.online:
                                print("not online")
                                time.sleep(0.1)
                            self.p.send_now(Command)

                    ins_id = ins_id + 1
                    if(runandload):
                        ins_count = prog.InstructionCount()
                    print("Instruction",ins_id,"/",ins_count,"  ",ins_nom)
                except SystemExit:
                    print("Killed")
                    sys.exit()
                except:
                    ins_id = ins_id + 1
                    print(ins_nom)
                    pass
            else:
                print('Paused at instruction: ',ins_id,"  " ,ins_nom)
                while self.app.paused:
                    time.sleep(1)


        home()
        self.app.do_settemp("0")
        self.p.send_now("M107 P0")
        self.p.send_now("M107 P1")
        app2.mainwindow.p.printing=False
        time.sleep(2)
        global task
        task= "Printing finished"
        quit()


def Pronterface():
    global app2, RDK, t1, t2 ,t3
    os.environ['GDK_BACKEND'] = 'x11'
    app2 = MyApp(False)
    app2.mainwindow.connect()
    app2.MainLoop()
    app2.mainwindow.do_settemp("0")
    app2.mainwindow.p.send_now("M107 P0")
    app2.mainwindow.p.send_now("M107 P1")
    wx.CallAfter(app2.mainwindow.pausebtn.SetLabel, _("&Pause"))
    wx.CallAfter(app2.mainwindow.pausebtn.Enable)
    time.sleep(2)
    print("Closing")
    del app2
    t2.kill()
    t2.join()
    home()
    if(runandload):
        t3.kill()
        t3.join()
    tracer.stop()
    tracer.save() # also takes output_file as an optional argument

def Safety(q):
    RDK2 = Robolink()
    CAM_NAME = "Camera 1"
    cam_item = RDK2.Item(CAM_NAME, ITEM_TYPE_CAMERA)
    if not cam_item.Valid():
        raise Exception("Camera not found! %s" % CAM_NAME)

    tempdir = tempfile.gettempdir()
    snapshot_file = tempdir + "\\sift_temp.png"
    print(snapshot_file)
    # Create a pipeline

    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    x=400
    y=300
    new_point=[x,y]
    points=[new_point]
    distances=[100]
    maxd=1800
    mind=500

    r=0

    with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1.8 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale
        clipping_distance=math.ceil(clipping_distance)
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)
        thre=rs.threshold_filter(0,clipping_distance_in_meters)
        hole_filling = rs.hole_filling_filter(1)
        colorizer = rs.colorizer()
        # spatial = rs.spatial_filter()
        spatial = rs.spatial_filter(0.5,20,2,4)
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.

        # Streaming loop
        try:
            while True:
                global image
                # Get frameset of color and depth
                frames = pipeline.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()
                aligned_depth_frame_thre=thre.process(aligned_depth_frame)

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                image = np.asanyarray(color_frame.get_data())
                image_width, image_height, _ = image.shape
                # Remove background - Set pixels further than clipping_distance to grey
                red_color = 0
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), red_color, image)

                # Render images:
                #   depth align to color on left
                #   depth on right
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
                images = np.hstack((bg_removed, depth_colormap))
                bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                bg_removed.flags.writeable = False

                # Process the image and find hands
                results = hands.process(bg_removed)

                bg_removed.flags.writeable = True

                # Draw the hand annotations on the image.
                bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_RGB2BGR)

                colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame_thre).get_data())

                filled_depth = hole_filling.process(aligned_depth_frame)
                filled_depth=thre.process(filled_depth)
                colorized_depthfiltered = np.asanyarray(colorizer.colorize(filled_depth).get_data())

                filled_depth2 = spatial.process(aligned_depth_frame)
                filled_depth2=thre.process(filled_depth2)
                colorized_depthfiltered2 = np.asanyarray(colorizer.colorize(filled_depth2).get_data())
                depthdata=np.asanyarray(filled_depth2.get_data())
                depths=depthdata[depthdata!=0]
                depths=depths[depths!=clipping_distance]

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        new_point1=int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_height),int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_width)
                        new_point2=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_height),int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_width)
                        new_point=int((new_point1[0]+new_point2[0])/2),int((new_point1[1]+new_point2[1])/2)
                        points.append(new_point)
                        mp_drawing.draw_landmarks(bg_removed, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        distances.append(depthdata[new_point[1], new_point[0]])
                        # print(new_point)
                    for i in range (len(points)):
                        cv2.circle(bg_removed,(points[i][0], points[i][1] ),2,(255,255,255),2)
                        cv2.putText(bg_removed, "{}mm".format(distances[i]),(points[i][0], points[i][1] ), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255,255), 2)
                        # cv2.putText(bg_removed, "{}mm".format(distances[i]), (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x) , int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)  ), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255,255), 2)

                images = np.hstack((bg_removed, colorized_depth,colorized_depthfiltered,colorized_depthfiltered))
                images = np.hstack((bg_removed,colorized_depthfiltered))
                # text="Min "+str(mind)+" Max "+str(maxd)
                # cv2.putText(img=images, text=text, org=(50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
                # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
                # cv2.imshow('Align Example', images)

                imagerob=bg_removed.copy()

                RDK2.Cam2D_Snapshot(snapshot_file, cam_item)
                img_scene = cv2.imread(snapshot_file, cv2.IMREAD_GRAYSCALE)
                img_scene=cv2.resize(img_scene,(640,480))
                (thresh, im_bw) = cv2.threshold(img_scene, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # print(image.shape)
                # print(im_bw.shape)
                kernel = np.ones((3,3),np.uint8)
                closing = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
                gradient = cv2.morphologyEx(im_bw, cv2.MORPH_GRADIENT, kernel)
                masked = cv2.bitwise_and(imagerob, imagerob, mask=closing)
                depthsmasked = cv2.bitwise_and(depthdata, depthdata, mask=closing)
                colorized_depthfilteredmasked=cv2.bitwise_and(colorized_depthfiltered,colorized_depthfiltered, mask=closing)
                depthnew=np.asanyarray(colorized_depthfilteredmasked)

                vis = np.concatenate((masked,colorized_depthfilteredmasked), axis=1)
                # cv2.imshow("Masked Robot",vis)
                # cv2.imshow("Morphology",np.concatenate((gradient,closing), axis=1))
                def pause2():
                    response = requests.get("http://localhost:5000/pause")
                def resume2():
                    response = requests.get("http://localhost:5000/resume")
                if points:
                    for p in points:
                        if depthdata[p[1],p[0]] < maxd and depthdata[p[1],p[0]] >mind:
                            newxy=np.where(gradient==255)
                            tarx=abs(newxy[0]-p[1])
                            tary=abs(newxy[1]-p[0])
                            tarz=abs(depthdata[newxy[0],newxy[1]]-depthdata[p[1],p[0]])
                            tar1=np.power(np.square(tarx)+np.square(tary)+np.square(tarz),0.5)
                            u=tar1[tar1<100]
                            if(len(u)!=0):
                                # print("Depth is:",depthdata[p[1],p[0]],"and max min is",mind,maxd)
                                cv2.putText(img=images, text="Collision Stop", org=(50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
                                if(r==0):
                                    t10 = Thread(name='Pause', target=pause2)
                                    t10.start()
                                    r=1
                                # print(response)
                                # task="Paused to avoid collision"
                                # print("stop")
                else:
                    depthsmaskedfil=depthsmasked[depthsmasked != 0]
                    maxd=np.max(depthsmaskedfil)
                    mind=np.min(depthsmaskedfil)-50
                    # print("resume")
                    if(r==1):
                        t11 = Thread(name='Resume', target=resume2)
                        t11.start()
                        r=0
                    # print(response)
                    # task=" "
                    # app2.mainwindow.pause(toggle=2)
                # print(safety)
                points.clear()
                distances.clear()
                cv2.imshow("Bg removed Robot1",images)
                #images
                global camera
                if camera:
                    try:
                        q.put(cv2.flip(image, 1),block=False)
                    except:
                        pass
                    key = cv2.waitKey(1)
                    # Press esc or 'q' to close the image window
                    if key & 0xFF == ord('q') or key == 27:
                        cv2.destroyAllWindows()
                        break
        except:
            pass
        finally:
            pipeline.stop()

if __name__ == "__main__":
    global t1, t2 ,t3,app, RDK, program, app2,job ,t5,t6,q
    q = Queue(maxsize=50)
    t1 = Thread(name='Pronterface', target=Pronterface)
    t1.start()
    job=ABBPrint()
    t5 = Thread(name='Run_Web', target=Run_Web)
    t5.start()
    t6 = Process(name='Safety', target=Safety,args=(q,))
    t6.start()
    job.StartPrint()
