"""Exactly three mutually exclusive painting ending conditions."""


def time_out(env, command_name="painting_command"):
    command = env.command_manager.get_term(command_name)
    command.refresh()
    return command.timeout


def bad_orientation(env, command_name="painting_command"):
    command = env.command_manager.get_term(command_name)
    command.refresh()
    return command.bad


def success(env, command_name="painting_command"):
    command = env.command_manager.get_term(command_name)
    command.refresh()
    return command.success
