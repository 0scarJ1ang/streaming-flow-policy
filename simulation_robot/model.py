import mujoco_py
import os

# 定义一个极简模型
xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3"/>
    </body>
  </worldbody>
</mujoco>
"""

# 尝试加载并编译模型
model = mujoco_py.load_model_from_xml(xml)
sim = mujoco_py.MjSim(model)

print("\n==============================")
print("SUCCESS: MuJoCo is fully working!")
print("==============================")