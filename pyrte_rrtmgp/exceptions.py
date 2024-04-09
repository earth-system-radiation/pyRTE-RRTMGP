class NotInternalSourceError(ValueError):
    pass


class NotExternalSourceError(ValueError):
    pass


class MissingAtmosfericConditionsError(AttributeError):
    message = (
        "You need to load the atmospheric conditions first."
        "Use the method load_atmosferic_conditions with an appropriated file."
    )

    def __init__(self, message=message):
        super().__init__(message)
