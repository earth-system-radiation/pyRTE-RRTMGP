class NotInternalSourceError(ValueError):
    pass


class NotExternalSourceError(ValueError):
    pass


class MissingAtmosphericConditionsError(AttributeError):
    message = (
        "You need to load the atmospheric conditions first."
        "Use the method load_atmospheric_conditions with an appropriated file."
    )

    def __init__(self, message=message):
        super().__init__(message)
