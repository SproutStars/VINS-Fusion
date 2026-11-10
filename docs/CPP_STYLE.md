# C++ 代码规范与检查

项目使用 Clang 18。根目录 `.clang-format` 为独立基准程序提供排版规则，
各 ROS 包中的 `.clang-format` 保留包级设置。根目录 `.clang-tidy` 定义共同的
静态检查规则，`vins_estimator/.clang-tidy` 继承根配置。

## 规则

- 缩进为 2 个空格，列宽为 110；函数、命名空间的左花括号换行。
- 类型使用大驼峰，结构体、枚举、联合体分别使用 `_s`、`_e`、`_u` 后缀。
- 方法使用小驼峰，参数使用 `_` 前缀，数据成员使用 `_` 后缀。
- 全局变量、枚举值与静态常量使用全大写。
- 重写方法使用 `override`，空构造/析构在语义允许时使用 `= default`。
- 未使用参数保留名称并标注 `[[maybe_unused]]`。
- 使用 `nullptr`、范围循环、`empty()` 等规则建议的写法。

`EnumSuffix` 接受单个字面后缀，因此原来的 `_e;_i` 统一为 `_e`。
`ClassMemberSuffix` 只检查静态成员；普通数据成员由 `MemberSuffix` 检查。

第三方源码不参与格式化，静态检查的头文件过滤器排除 ThirdParty 目录。
Ceres `AutoDiffManifold` 要求使用 `Plus`、`Minus` 方法名，这些位置使用带说明的
局部 `NOLINTNEXTLINE`。枚举保留原来的底层表示，并局部说明为何不缩小其存储类型。
浮点数学调用显式保留原来的 double 转换，以免在规范整理中改变数值精度。

## 执行检查

在仓库根目录执行：

```sh
sudo apt install clang-format-18 clang-tidy-18
source /opt/ros/jazzy/setup.bash
colcon build --executor sequential --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTING=ON
source install/setup.bash
python3 scripts/check_cpp_style.py format
python3 scripts/check_cpp_style.py tidy --jobs 2
colcon test --executor sequential
colcon test-result --verbose
python3 -m pytest -q scripts/tests
```

`format` 覆盖 `src/` 与 `scripts/` 中的自有 C++ 文件，`tidy` 覆盖各包编译数据库中的
自有编译单元及其项目头文件，所有可见告警按错误处理。独立基准程序和未被任何编译单元
包含的头文件需要另外使用相应编译参数执行 clang-tidy。

修复排版可执行 `python3 scripts/check_cpp_style.py format --fix`。
工具路径可由 `CLANG_FORMAT`、`CLANG_TIDY` 环境变量指定，构建目录可用
`--build-dir` 指定。静态检查不提供批量自动修复选项；跨编译单元重命名应先汇总修改位置，
再统一修改声明、定义、模板调用和测试。

此次重构改变了部分 C++ 类型、方法和成员名称。下游 C++ 调用代码需要同步使用新名称，
并重新构建工作区。

## 本次验证

Schur 后端独立基准使用相同的 `-std=c++17 -O2 -DNDEBUG` 参数，
重构前后得到完全一致的两项校验值：

- reduce：`446728580.03886902`
- update：`12808.410299953161`

这些校验覆盖固定合成场景，不代表完整数据集的轨迹精度评测。
