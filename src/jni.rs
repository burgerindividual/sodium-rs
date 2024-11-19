use core::marker::PhantomData;
use core::mem::transmute_copy;

pub type JEnv = core::ffi::c_void;
pub type JClass = core::ffi::c_void;

pub type Jbyte = i8;
pub type Jshort = i16;
pub type Jint = i32;
pub type Jlong = i64;

pub type Jfloat = f32;
pub type Jdouble = f64;

pub type Jchar = u16;

pub type Jboolean = bool;

#[repr(transparent)]
pub struct JPtr<T> {
    addr: Jlong,
    _type: PhantomData<T>,
}

impl<T> JPtr<T> {
    /// SAFETY: The pointer must be of type T
    pub unsafe fn as_ref(&self) -> &T {
        let ptr = self.as_ptr();
        ptr.as_ref().expect("ptr must not be null")
    }

    /// SAFETY: The pointer must be of type T
    pub unsafe fn as_ptr(&self) -> *const T {
        self.addr as *const T
    }
}

impl<T> From<*const T> for JPtr<T> {
    fn from(value: *const T) -> Self {
        Self {
            addr: value as usize as Jlong,
            _type: Default::default(),
        }
    }
}

impl<T> From<&T> for JPtr<T> {
    fn from(value: &T) -> Self {
        (value as *const T).into()
    }
}

#[repr(transparent)]
pub struct JPtrMut<T> {
    addr: Jlong,
    _type: PhantomData<T>,
}

impl<T> JPtrMut<T> {
    /// SAFETY: The pointer must be of type T
    /// SAFETY: The backing memory of the pointer must allow mutating
    /// SAFETY: This consumes the JPtrMut instance to uphold borrow checker
    /// rules
    pub unsafe fn into_mut_ref<'a>(self) -> &'a mut T {
        self.into_mut_ptr().as_mut().expect("ptr must not be null")
    }

    /// SAFETY: The pointer must be of type T
    /// SAFETY: The backing memory of the pointer must allow mutating
    /// SAFETY: This consumes the JPtrMut instance to uphold borrow checker
    /// rules
    pub unsafe fn into_mut_ptr(self) -> *mut T {
        self.addr as *mut T
    }
}

impl<T> From<*mut T> for JPtrMut<T> {
    fn from(value: *mut T) -> Self {
        Self {
            addr: value as usize as Jlong,
            _type: Default::default(),
        }
    }
}

impl<T> From<&mut T> for JPtrMut<T> {
    fn from(value: &mut T) -> Self {
        (value as *mut T).into()
    }
}

#[repr(transparent)]
pub struct JFnPtr<T> {
    addr: Jlong,
    _type: PhantomData<T>,
}

impl<T> JFnPtr<T> {
    /// SAFETY: The pointer must be of type T
    pub unsafe fn as_fn_ptr(&self) -> Option<T> {
        if self.addr == 0 {
            return None;
        }

        Some(transmute_copy(&(self.addr as usize)))
    }
}
