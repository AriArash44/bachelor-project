import { useContext, useRef, useEffect } from 'react';
import { ModalContext } from '../contexts/modalContext.js';

export default function Modal({ children }) {
  const { isOpen, closeModal } = useContext(ModalContext);
  const modalRef = useRef();
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        closeModal();
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, closeModal]);
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: "rgba(0, 0, 0, 0.5)" }} >
      <div
        ref={modalRef}
        className="bg-white rounded-2xl shadow-lg p-8 pt-18 pr-12 relative animate-fadeIn"
      >
        {children}
      </div>
    </div>
  );
}
