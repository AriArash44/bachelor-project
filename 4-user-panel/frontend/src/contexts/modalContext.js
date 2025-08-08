import { createContext, useState, useContext } from 'react';

const ModalContext = createContext(false);

function useModalState() {
  const [isOpen, setIsOpen] = useState(false);
  const openModal = () => setIsOpen(true);
  const closeModal = () => setIsOpen(false);
  return { isOpen, openModal, closeModal };
}

function useModal() {
  return useContext(ModalContext);
}

export { ModalContext, useModalState, useModal };
