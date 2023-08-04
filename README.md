check model path, topic name, gpu memory   
docker image is in Nas120/TrafficLight/4_code/AAP_v0.9.1/211102_AAP_v0.9.1 + trafficlight_code(final)/docker/tl.tar   


how to run    
(Trafficlight_inference_ws)$ catkin_make   
(Trafficlight_inference_ws)$ source devel/setup.bash   
(Trafficlight_inference_ws)$ python src/AAP_trafficlight/src/main.py   

[Troubleshooting]
 - node 실행 시, ‘python\r’이 들어가 실행이 안되는 경우
dgist@stillrunning-IP238:~/catkin_ws$ rosrun dgist_trafficlight inference.py
/usr/bin/env: ‘python\r’: No such file or directory
 * 파일 복사 등에 의해 \r이 추가되어 실행되지 않는 문제이므로 해당 문제 삭제 후 실행

 - 모델 실행 시, "AttributeError: 'module' object has no attribute 'Str'"가 발생
WARNING:tensorflow:Entity <bound method MaxPooling2D.call of <tensorflow.python.layers.pooling.MaxPooling2D object at 0x7fe971949a50>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method MaxPooling2D.call of <tensorflow.python.layers.pooling.MaxPooling2D object at 0x7fe971949a50>>: AttributeError: 'module' object has no attribute 'Str'
WARNING:tensorflow:Entity <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x7fe92c095910>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x7fe92c095910>>: AttributeError: 'module' object has no attribute 'Index'
 * 대부분이 PIP gast의 version이 0.2.2아니라서 생기는 경우라 하기와 같이 버전 변경하면 해결될 수 있음.
 	sudo pip2 install -U gast==0.2.2

 - 모델 실행 시, "[rosrun] or not executable"가 발생
dgist@stillrunning-IP238:~/catkin_ws$ rosrun dgist_trafficlight inference.py
[rosrun] Couldn't find executable named inference.py below /home/dgist/catkin_ws/src/dgist_trafficlight
[rosrun] Found the following, but they're either not files,
[rosrun] or not executable:
[rosrun]   /home/dgist/catkin_ws/src/dgist_trafficlight/src/inference.py
 * 대부분이 파일 권한 문제로 발생하므로 하기 명령어로 해결가능할 수 있음.
    sudo chmod +x /home/dgist/catkin_ws/src/dgist_trafficlight/src/inference.py

 - Xavier에서 모델 실행 시, numpy 관련 에러 발생
ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
ImportError: numpy.core.multiarray failed to import
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "<frozen importlib._bootstrap>", line 980, in _find_and_load
SystemError: <class '_frozen_importlib._ModuleLockManager'> returned a result with an error set
ImportError: numpy.core._multiarray_umath failed to import
ImportError: numpy.core.umath failed to import
2021-11-16 10:58:52.953023: F tensorflow/python/lib/core/bfloat16.cc:675] Check failed: PyBfloat16_Type.tp_base != nullptr 
 * 대부분이 numpy version 문제로 하기 명령어로 해결가능할 수 있음.
      sudo pip3 install -U numpy==1.16.3