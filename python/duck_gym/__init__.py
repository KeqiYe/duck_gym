"""Native CPU/CUDA simulation runtimes, loaded only for the selected backend."""
__all__ = ['StandingEnv']

def __getattr__(name):
    if name == 'StandingEnv':
        from .env import StandingEnv
        return StandingEnv
    raise AttributeError(name)
