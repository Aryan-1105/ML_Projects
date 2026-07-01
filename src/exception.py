import sys  # Provides access to system-specific parameters and exception details
import logging

def error_message_detail(error, error_detail: sys):
    """
    Creates a detailed error message containing:
    - File name
    - Line number
    - Original error message
    """

    # Get exception information
    # exc_info() returns (exception_type, exception_value, traceback)
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a detailed error message
    error_message = (
        f"Error occurred in Python script [{file_name}] "
        f"at line number [{exc_tb.tb_lineno}] "
        f"Error message: {str(error)}"
    )

    return error_message


# Custom exception class
# Inherits all properties of Python's built-in Exception class
class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys):

        # Call the parent Exception class constructor
        super().__init__(error_message)

        # Store the detailed error message
        self.error_message = error_message_detail(error_message, error_detail)

    # This method is automatically called when we print the exception
    def __str__(self):
        return self.error_message

