import sys


def error_message_detail(error, error_detail: sys):
    """
    Creates a detailed error message containing:
    1. File name
    2. Line number
    3. Original error message

    Parameters
    ----------
    error : Exception
        The exception that occurred.

    error_detail : sys
        The sys module used to extract traceback information.

    Returns
    -------
    str
        A formatted error message.
    """

    # Extract traceback information
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a detailed error message
    error_message = (
        f"Error occurred in Python script: [{file_name}] "
        f"at line number [{exc_tb.tb_lineno}] "
        f"with error message: [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom Exception Class

    This class extends Python's built-in Exception class
    and provides detailed error information.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the custom exception.

        Parameters
        ----------
        error_message : Exception
            Original exception.

        error_detail : sys
            Python sys module.
        """

        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

    def __str__(self):
        """
        Return the formatted error message.
        """
        return self.error_message