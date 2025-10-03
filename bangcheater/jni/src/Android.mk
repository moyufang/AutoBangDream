LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := bangcheater

LOCAL_SRC_FILES := \
    bangcheater.c

LOCAL_STATIC_LIBRARIES := \
    libevdev

include $(BUILD_EXECUTABLE)