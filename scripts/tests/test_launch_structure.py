"""Unit tests for the VINS-Fusion launch entry points.

The tests use the real ROS 2 launch and launch_ros Python modules with a
LaunchContext, but they never start nodes, RViz, or other processes.
"""
import importlib.util
import os
from pathlib import Path
import unittest

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.utilities.evaluate_parameters import evaluate_parameters

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DIR = REPO_ROOT / "src" / "vins_estimator" / "launch"
VINS_SHARE = Path(FindPackageShare("vins").perform(LaunchContext()))

LEGACY_ENTRIES = {
    "vins_mono_imu_launch.py": "mono_imu",
    "vins_stereo_imu_launch.py": "stereo_imu",
    "vins_stereo_launch.py": "stereo",
}
MODES = ("mono_imu", "stereo_imu", "stereo")


def load_launch_description(filename):
    path = LAUNCH_DIR / filename
    spec = importlib.util.spec_from_file_location(
        "vins_launch_test_" + path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load launch file: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def declarations(description):
    return [
        entity for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    ]


def declaration(description, name):
    matches = [item for item in declarations(description) if item.name == name]
    if len(matches) != 1:
        raise AssertionError(
            "expected exactly one {!r} declaration, got {}".format(name, len(matches)))
    return matches[0]


def apply_declarations(description, overrides=None):
    """Execute declarations in order against a real LaunchContext."""
    context = LaunchContext()
    if overrides:
        context.launch_configurations.update(overrides)
    for item in declarations(description):
        item.execute(context)
    return context


def mode_config_path(mode):
    return os.path.normpath(os.path.join(
        str(VINS_SHARE), "config", "euroc",
        "euroc_{}_config.yaml".format(mode)))


def nodes_by_executable(description):
    return {
        entity.node_executable: entity
        for entity in description.entities
        if isinstance(entity, Node)
    }


def evaluated_node_parameters(node, context):
    parameters = node._Node__parameters
    evaluated = evaluate_parameters(context, parameters)
    if len(evaluated) != 1 or not isinstance(evaluated[0], dict):
        raise AssertionError("expected one parameter dictionary for {}".format(
            node.node_executable))
    return evaluated[0]


class VinsLaunchStructureTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.description = load_launch_description("vins.launch.py")

    def test_default_mode_and_existing_defaults(self):
        context = apply_declarations(self.description)
        configurations = context.launch_configurations

        self.assertEqual(configurations["mode"], "mono_imu")
        self.assertEqual(
            os.path.normpath(configurations["config_file"]),
            mode_config_path("mono_imu"))
        self.assertEqual(configurations["use_sim_time"], "false")
        self.assertEqual(configurations["loop_fusion"], "true")
        self.assertEqual(configurations["global_fusion"], "false")
        self.assertEqual(configurations["rviz"], "true")
        self.assertEqual(configurations["opencv_threads"], "1")
        self.assertEqual(
            os.path.normpath(configurations["output_path"]),
            os.path.normpath(os.path.join(str(VINS_SHARE), "..", "..", "output")))

        vins_node = nodes_by_executable(self.description)["vins_node"]
        parameters = evaluated_node_parameters(vins_node, context)
        self.assertEqual(
            os.path.normpath(parameters["config_file"]),
            mode_config_path("mono_imu"))
        self.assertEqual(
            os.path.normpath(parameters["output_path"]),
            os.path.normpath(os.path.join(str(VINS_SHARE), "..", "..", "output")))
        self.assertIs(parameters["use_sim_time"], False)
        self.assertEqual(parameters["opencv_threads"], 1)
        self.assertIsInstance(parameters["opencv_threads"], int)

    def test_mode_declaration_has_expected_choices(self):
        mode = declaration(self.description, "mode")
        self.assertEqual(mode.choices, list(MODES))
        self.assertEqual(
            perform_substitutions(LaunchContext(), mode.default_value),
            "mono_imu")

    def test_each_mode_selects_its_euroc_config(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                context = apply_declarations(self.description, {"mode": mode})
                self.assertEqual(
                    os.path.normpath(context.launch_configurations["config_file"]),
                    mode_config_path(mode))

                vins_node = nodes_by_executable(self.description)["vins_node"]
                parameters = evaluated_node_parameters(vins_node, context)
                self.assertEqual(
                    os.path.normpath(parameters["config_file"]),
                    mode_config_path(mode))

    def test_explicit_config_file_overrides_mode_default(self):
        custom_config = "/tmp/vins_custom_config.yaml"
        for mode in MODES:
            with self.subTest(mode=mode):
                context = apply_declarations(
                    self.description,
                    {"mode": mode, "config_file": custom_config})
                self.assertEqual(
                    context.launch_configurations["config_file"], custom_config)

                vins_node = nodes_by_executable(self.description)["vins_node"]
                parameters = evaluated_node_parameters(vins_node, context)
                self.assertEqual(parameters["config_file"], custom_config)

    def test_existing_node_names_and_conditions(self):
        context = apply_declarations(self.description)
        nodes = nodes_by_executable(self.description)

        self.assertEqual(set(nodes), {
            "vins_node", "loop_fusion_node", "global_fusion_node", "rviz2"})
        self.assertEqual(nodes["vins_node"]._Node__node_name, "vins_estimator")
        self.assertEqual(nodes["loop_fusion_node"]._Node__node_name, "loop_fusion")
        self.assertEqual(nodes["global_fusion_node"]._Node__node_name, "global_fusion")
        self.assertIsNone(nodes["vins_node"].condition)
        self.assertIsInstance(nodes["loop_fusion_node"].condition, IfCondition)
        self.assertIsInstance(nodes["global_fusion_node"].condition, IfCondition)
        self.assertIsInstance(nodes["rviz2"].condition, IfCondition)
        self.assertTrue(nodes["loop_fusion_node"].condition.evaluate(context))
        self.assertFalse(nodes["global_fusion_node"].condition.evaluate(context))
        self.assertTrue(nodes["rviz2"].condition.evaluate(context))

        enabled = apply_declarations(
            self.description,
            {"loop_fusion": "true", "global_fusion": "true", "rviz": "false"})
        self.assertTrue(nodes["loop_fusion_node"].condition.evaluate(enabled))
        self.assertTrue(nodes["global_fusion_node"].condition.evaluate(enabled))
        self.assertFalse(nodes["rviz2"].condition.evaluate(enabled))

    def test_primary_odometry_routes_only_corrected_output_when_loop_enabled(self):
        nodes = nodes_by_executable(self.description)
        for enabled in ("true", "false"):
            description = load_launch_description("vins.launch.py")
            nodes = nodes_by_executable(description)
            context = apply_declarations(description, {"loop_fusion": enabled})
            expected = "/vins_estimator/odometry_raw" if enabled == "true" else "/vins_estimator/odometry"
            for executable, source in (("vins_node", "~/odometry"),
                                       ("global_fusion_node", "/vins_estimator/odometry")):
                remaps = {perform_substitutions(context, a): perform_substitutions(context, b)
                          for a, b in nodes[executable]._Node__remappings}
                self.assertEqual(remaps[source], expected)
            loop = nodes["loop_fusion_node"]
            self.assertEqual(loop.condition.evaluate(context), enabled == "true")
            params = evaluated_node_parameters(loop, context)
            self.assertEqual(params["input_odometry_topic"], "/vins_estimator/odometry_raw")
            self.assertTrue(params["publish_primary_odometry"])
            self.assertTrue(params["publish_primary_visualization"])
            estimator = nodes["vins_node"]
            self.assertEqual(evaluated_node_parameters(estimator, context)["use_raw_frames"],
                             enabled == "true")
            remaps = {perform_substitutions(context, a): perform_substitutions(context, b)
                      for a, b in estimator._Node__remappings}
            self.assertEqual(remaps["~/path"], "/vins_estimator/path_raw" if enabled == "true"
                             else "/vins_estimator/path")

    def test_fusion_output_parameters_are_unchanged(self):
        context = apply_declarations(
            self.description, {"loop_fusion": "true", "global_fusion": "true"})
        nodes = nodes_by_executable(self.description)
        output = os.path.normpath(context.launch_configurations["output_path"])

        loop_parameters = evaluated_node_parameters(nodes["loop_fusion_node"], context)
        self.assertEqual(
            os.path.normpath(loop_parameters["pose_graph_save_path"]),
            os.path.normpath(os.path.join(output, "pose_graph")))

        global_parameters = evaluated_node_parameters(
            nodes["global_fusion_node"], context)
        self.assertEqual(
            os.path.normpath(global_parameters["output_file"]),
            os.path.normpath(os.path.join(output, "vio_global.csv")))


class LegacyLaunchEntryTest(unittest.TestCase):

    def test_legacy_entries_only_forward_the_mode(self):
        for filename, expected_mode in LEGACY_ENTRIES.items():
            with self.subTest(launch_file=filename):
                description = load_launch_description(filename)
                self.assertFalse(any(
                    isinstance(entity, DeclareLaunchArgument)
                    for entity in description.entities))
                self.assertEqual(len(description.entities), 1)

                include = description.entities[0]
                self.assertIsInstance(include, IncludeLaunchDescription)
                self.assertIsInstance(
                    include.launch_description_source,
                    PythonLaunchDescriptionSource)

                context = LaunchContext()
                # Executing the include expands its source path without visiting
                # the returned nested launch description.
                included_entities = include.execute(context)
                self.assertEqual(len(included_entities), 2)
                self.assertEqual(
                    Path(include.launch_description_source.location).name,
                    "vins.launch.py")

                forwarded = {
                    perform_substitutions(
                        context, normalize_to_list_of_substitutions(name)):
                    perform_substitutions(
                        context, normalize_to_list_of_substitutions(value))
                    for name, value in include.launch_arguments
                }
                self.assertEqual(forwarded, {"mode": expected_mode})

    def test_cli_overrides_survive_legacy_forwarding(self):
        custom_config = "/tmp/vins_legacy_override.yaml"
        for filename, expected_mode in LEGACY_ENTRIES.items():
            with self.subTest(launch_file=filename):
                description = load_launch_description(filename)
                include = description.entities[0]

                context = LaunchContext()
                context.launch_configurations["config_file"] = custom_config
                context.launch_configurations["use_sim_time"] = "true"

                included_entities = include.execute(context)
                nested_descriptions = [
                    entity for entity in included_entities
                    if isinstance(entity, LaunchDescription)
                ]
                self.assertEqual(len(nested_descriptions), 1)
                for entity in included_entities:
                    if isinstance(entity, SetLaunchConfiguration):
                        entity.execute(context)

                # The nested launch declarations run after the forwarded mode is
                # set and must not replace the command-line overrides.
                for entity in nested_descriptions[0].entities:
                    if isinstance(entity, DeclareLaunchArgument):
                        entity.execute(context)

                self.assertEqual(
                    context.launch_configurations["mode"], expected_mode)
                self.assertEqual(
                    context.launch_configurations["config_file"], custom_config)
                self.assertEqual(
                    context.launch_configurations["use_sim_time"], "true")


if __name__ == "__main__":
    unittest.main()
