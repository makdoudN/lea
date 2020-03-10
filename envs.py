from gym.wrappers.rescale_action import RescaleAction
from rltk.common.utils import import_class_from_string
from gym.spaces.box import Box
import gym


def make_env(name: str, loader: str = "gym", **kwargs):
    rescale_action = kwargs.get('rescale_action', True)
    env = None
    if loader == "gym":     # Base Wrapper
        try:
            from gym_extensions.continuous import mujoco
        except:
            print('gym_extensions import failure !')
            pass
        env = gym.make(name)
    elif loader == "metaworld":
        env = import_class_from_string(name)(**kwargs)
    elif loader == "dm_control":
        import dmc2gym
        # Useful Options: frame_skip, from_pixels
        # Note: dmc2gym normalized the action but still we can
        # use RescaleAction.
        # NOTE: in the future maybe simply return the env.
        env = dmc2gym.make(domain_name=name, **kwargs)
    elif loader == 'atari':
        raise NotImplementedError()

    assert env is not None
    if isinstance(env.action_space, Box) and rescale_action:
        # Environment has continuous space and by default is normalized.
        env = RescaleAction(env, -1, 1)
    return env
