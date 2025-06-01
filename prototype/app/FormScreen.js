import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, Alert } from 'react-native';
import config from './config';

// Pantalla donde el usuario introduce su información personal
export default function FormScreen({ navigation }) {
  const [formData, setFormData] = useState({
    nombre: '',
    edad: '',
    lugarNacimiento: '',
    familiares: '',
    gustos: '',
  });

  // Maneja cambios en los campos del formulario
  const handleChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  // Envia los datos al backend y navega al chat
  const handleSubmit = async () => {
    const url = config.BASE_URL;

    try {
      const response = await fetch(`${url}/setup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Error ${response.status}: ${text}`);
      }

      const data = await response.json();

      Alert.alert('Datos enviados', 'Tus datos se han registrado correctamente.');
      // Redirige al chat con el mensaje de bienvenida generado
      navigation.navigate('Chat', { welcome_msg: data.welcome_msg });
    } catch (error) {
      console.error('Error enviando datos', error);
      Alert.alert('Error', 'No se pudieron enviar los datos.');
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.label}>Nombre y apellidos</Text>
      <TextInput
        style={styles.input}
        value={formData.nombre}
        onChangeText={text => handleChange('nombre', text)}
        placeholder="Samuel Sánchez Carrasco"
      />

      <Text style={styles.label}>Edad</Text>
      <TextInput
        style={styles.input}
        value={formData.edad}
        onChangeText={text => handleChange('edad', text)}
        placeholder="75"
        keyboardType="numeric"
      />

      <Text style={styles.label}>Lugar de nacimiento</Text>
      <TextInput
        style={styles.input}
        value={formData.lugarNacimiento}
        onChangeText={text => handleChange('lugarNacimiento', text)}
        placeholder="Torredonjimeno"
      />

      <Text style={styles.label}>Familiares cercanos</Text>
      <TextInput
        style={styles.input}
        value={formData.familiares}
        onChangeText={text => handleChange('familiares', text)}
        placeholder="Rafael e Isabel (padres); Adrián (hermano)"
      />

      <Text style={styles.label}>Gustos personales</Text>
      <TextInput
        style={styles.input}
        value={formData.gustos}
        onChangeText={text => handleChange('gustos', text)}
        placeholder="Leer, escuchar música, pasear..."
      />

      <View style={styles.buttonContainer}>
        <Button title="Enviar y chatear" onPress={handleSubmit} color="#128c7e" />
      </View>
    </ScrollView>
  );
}

// Estilos
const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: '#fff',
    flexGrow: 1,
  },
  label: {
    fontSize: 16,
    marginTop: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 10,
    borderRadius: 8,
    marginTop: 5,
  },
  buttonContainer: {
    marginTop: 20,
  },
});
