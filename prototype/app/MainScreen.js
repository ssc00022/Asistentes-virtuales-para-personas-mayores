import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, FlatList, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native';
import { Audio } from 'expo-av';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import config from './config';

// Pantalla principal del asistente de voz
export default function MainScreen({ route }) {
  const [recording, setRecording] = useState(null); // Objeto de grabación actual
  const [messages, setMessages] = useState([]); // Historial de mensajes (enviados y recibidos)
  const [isRecording, setIsRecording] = useState(false); // Estado de grabación
  const [sound, setSound] = useState(null); // Sonido actual a reproducir
  const flatListRef = useRef(null); // Referencia para autoscroll del chat
  const url = config.BASE_URL;
  const { welcome_msg } = route.params || {};

  // Al cargar el componente, mostrar el mensaje de bienvenida si existe
  useEffect(() => {
    if (welcome_msg) {
      setMessages([{ type: 'received', text: welcome_msg }]);
      fetchResponseAudio();
    }
  }, []);

  // Liberar recursos de sonido cuando cambie
  useEffect(() => {
    return sound ? () => sound.unloadAsync() : undefined;
  }, [sound]);

  // Scroll automático al nuevo mensaje
  useEffect(() => {
    if (flatListRef.current && messages.length > 0) {
      flatListRef.current.scrollToEnd({ animated: true });
    }
  }, [messages]);

  // Iniciar grabación de audio
  const startRecording = async () => {
    try {
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY
      );

      setRecording(recording);
      setIsRecording(true);
    } catch (err) {
      console.error('Error al iniciar la grabación', err);
    }
  };

  // Detener grabación y subir el archivo
  const stopRecording = async () => {
    setIsRecording(false);
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    setRecording(null);

    if (uri) {
      uploadAudio(uri);
    }
  };

  // Subir el audio grabado al servidor
  const uploadAudio = async (uri) => {
    const formData = new FormData();
    formData.append('file', {
      uri,
      name: 'audio.wav',
      type: 'audio/wav',
    });

    try {
      const response = await fetch(`${url}/receive`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Error ${response.status}: ${text}`);
      }
      const data = await response.json();

      setMessages(prev => [
        ...prev,
        { type: 'sent', text: data.transcription },
        { type: 'received', text: data.response }
      ]);

      fetchResponseAudio();
    } catch (err) {
      console.error('Error al subir el audio', err);
    }
  };

  // Obtener y reproducir el audio generado por el servidor
  const fetchResponseAudio = async () => {
    try {
      const response = await fetch(`${url}/audio`);
      if (response.ok) {
        const { sound: newSound } = await Audio.Sound.createAsync({ uri: `${url}/audio` });
        setSound(newSound);
        await newSound.playAsync();
      }
    } catch (error) {
      console.error('Error al reproducir el audio de respuesta', error);
    }
  };

  // Renderiza un mensaje en la conversación
  const renderMessage = ({ item }) => (
    <View style={[styles.messageBubble, item.type === 'received' ? styles.received : styles.sent]}>
      <Text style={styles.messageText}>{item.text}</Text>
    </View>
  );

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.select({ ios: 'padding', android: undefined })}
    >
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(_, index) => index.toString()}
      />

      <TouchableOpacity
        style={[styles.recordButton, isRecording ? styles.recording : null]}
        onPressIn={startRecording}
        onPressOut={stopRecording}
      >
        <MaterialCommunityIcons name="microphone" size={36} color="#fff" />
      </TouchableOpacity>
    </KeyboardAvoidingView>
  );
}

// Estilos de la pantalla
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    justifyContent: 'flex-end',
  },
  messageBubble: {
    margin: 8,
    padding: 12,
    borderRadius: 12,
    maxWidth: '75%',
  },
  received: {
    alignSelf: 'flex-start',
    backgroundColor: '#e0e0e0',
  },
  sent: {
    alignSelf: 'flex-end',
    backgroundColor: '#4e8cff',
  },
  messageText: {
    fontSize: 16,
    color: '#000',
  },
  recordButton: {
    backgroundColor: '#4e8cff',
    padding: 18,
    borderRadius: 50,
    alignSelf: 'center',
    margin: 16,
  },
  recording: {
    backgroundColor: '#d32f2f',
  },
});
