# AI3603_project
AI3603_project legged_gym_experiment

## DONE
-  环境配置
-  将参数命令行化，可以直接在命令行中设置主要的部分参数，同时将设定的参数打印到 log 中

## TODO
- [ ] 调参，主要方向：reward、algorithm、policy
- [ ] 尝试将测试标准规范化为数值表示，可以的话plot出来
- [ ] 调整测试环境为最后评估环境

## Problems
1. 躯干速度 v_trunk 是怎么评价的
2. 在敏捷性和稳定性的测量中的速度条件似乎没有给全
3. 在稳定性测量中加速度的定义是什么，code 中似乎没有定义，那需要给出粒度？
4. dof 中三种驱动模式和结果的对应关系
